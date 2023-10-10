#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <openmpi/ompi/mpi/cxx/mpicxx.h>
#include <random>
#include <typeinfo>
#include <vector>

#define DEBUG(x) x
#define STATUS(x) cout << "[rank " << RANK << "]: " << x;

#ifdef DEBUG
#include "./dbg.h"
#else
#define dbg(...)
#endif // DEBUG

#define ASSERT(x, msg)                                                         \
  if (!(x)) {                                                                  \
    cout << "ASSERTION FAILED: " << msg << endl;                               \
    exit(1);                                                                   \
  }
using namespace std;
using namespace MPI;
const double WEIGHT = 1;
const double INTERVAL = 1;
const int MAIN = 1;
int RANK;
int SIZE;
int N;
int D;

void coord_print(double *c) {
  cout << "(";
  for (int i = 0; i < D; i++) {
    cout << c[i] << ",";
  }
  cout << ")";
}

std::string coord_str(double *c) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < D; i++) {
    ss << std::scientific << std::setprecision(2) << c[i];
    if (i < D - 1) {
      ss << ",";
    }
  }
  ss << ")";
  return ss.str();
}

std::ostream &operator<<(std::ostream &out, double *v) {
  out << "(";
  for (int i = 0; i < D; i++) {
    out << v[i] << ",";
  }
  out << ")";
  return out;
}

struct Generator {
  mt19937 engine;
  normal_distribution<double> nd;
  discrete_distribution<int> dd;
  vector<vector<double>> means;
  vector<double> sds;

  Generator(unsigned long seed, ifstream &file) {
    int c;
    /* random_device rd{}; */
    /* gen = mt19937(rd()); */
    engine.seed(seed);
    nd = normal_distribution<double>(0, 1);
    file >> c;
    means = vector<vector<double>>(c, vector<double>(D));
    sds = vector<double>(c);
    vector<double> probs(c);

    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < D; ++j) {
        file >> means[i][j];
      }
      file >> sds[i];
      file >> probs[i];
    }
    dd = discrete_distribution<int>(probs.begin(), probs.end());
  }

  double normal() { return nd(engine); }
  double discrete() { return dd(engine); }
  double gaussian(double mean, double sd) { return normal() * sd + mean; }
  void rand_coord(double *p) {
    int i = dd(engine);
    for (int j = 0; j < D; j++) {
      p[j] = nd(engine) * sds[i] + means[i][j];
    }
  }
};

inline double *crd_at(double *arr, int i) { return arr + i * D; }

inline void crd_copy(double *src, double *dst) {
  memcpy(dst, src, D * sizeof(double));
}

double coord_dist_square(double *p1, double *p2) {
  double dist = 0;
  for (int i = 0; i < D; i++) {
    dist += pow(p1[i] - p2[i], 2);
  }
  return dist;
}

int coord_closest(double *p, double *centroids) {
  double min_dist = coord_dist_square(p, centroids);
  int min_idx = 0;
  for (int i = 1; i < SIZE; i++) {
    double *c = crd_at(centroids, i);
    double dist = coord_dist_square(p, c);
    if (dist < min_dist) {
      min_dist = dist;
      min_idx = i;
    }
  }
  return min_idx;
}

int main(int argc, char *argv[]) {
  // init
  MPI::Init();
  RANK = COMM_WORLD.Get_rank();
  SIZE = COMM_WORLD.Get_size();

  // all read file avoid communication
  ifstream file = ifstream(argv[1]);
  file >> N >> D;
  Generator gen(RANK, file);

  // init cluster
  int n = N; // number of mass
  double *centroids = new double[SIZE * D];
  double *positions = new double[N * D]; // over allocate avoid realloc
  double *velocities = new double[N * D];
  double *psum = new double[D];
  for (int i = 0; i < N; i++) {
    gen.rand_coord(crd_at(positions, i));
  }

  // first centroid
  uniform_int_distribution<int> uni(0, n - 1);
  int first = uni(gen.engine);
  double *p = crd_at(positions, first);
  /* COMM_WORLD.Bcast(centroids, D, MPI_DOUBLE, MAIN); */
  COMM_WORLD.Allgather(p, D, MPI_DOUBLE, centroids, D, MPI_DOUBLE);

  //
  // prepare for alltoallv
  //
  int *sendcounts = new int[SIZE];
  int *sdispls = new int[SIZE];
  int *recvcounts = new int[SIZE];
  int *rdispls = new int[SIZE];

  // sort centroids by distance
  double *sendbuf = new double[n * D];
  int *target_centroids = new int[n]; // tmp array for cal sendbuf offset
  memset(sendcounts, 0, SIZE * sizeof(int));
  for (int i = 0; i < n; i++) {
    int target = coord_closest(crd_at(positions, i), centroids);
    target_centroids[i] = target;
    sendcounts[target]++;
  }
  for (int i = 0; i < SIZE; i++) {
    cout << "rank " << RANK << "send to " << i
         << " sendcounts: " << sendcounts[i] << endl;
  }
  // cal prefix sum of sendcounts
  // for sendbuf offset
  sdispls[0] = 0;
  for (int i = 1; i < SIZE; i++) {
    sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
  }
  // for sendbuf[target_centroids[i]] offset
  memset(rdispls, 0, SIZE * sizeof(int));

  for (int i = 0; i < n; i++) {
    double *from = crd_at(positions, i);
    int t = target_centroids[i];
    int offset = sdispls[t] + rdispls[t];
    rdispls[t]++;
    double *to = crd_at(sendbuf, offset);
    cout << i << " at " << offset << endl;
    crd_copy(from, to);
  }

  COMM_WORLD.Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT);
  for (int i = 0; i < SIZE; i++) {
    cout << "rank " << RANK << " sdis[" << i << "]" << sdispls[i] << " rdis["
         << i << "]" << rdispls[i] << endl;
  }
  COMM_WORLD.Alltoall(sdispls, 1, MPI_INT, rdispls, 1, MPI_INT);
  for (int i = 0; i < SIZE; i++) {
    cout << "rank " << RANK << " to " << i << " sendcounts: " << sendcounts[i]
         << " senddispls " << sdispls[i] << rdispls[i] << endl;
  }

  MPI_Finalize();
  return 0;
}
