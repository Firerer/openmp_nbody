#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <openmpi/ompi/mpi/cxx/mpicxx.h>
#include <random>
#include <typeinfo>
#include <vector>

#define DEBUG(x) x
#define STATUS(x) cout << "[rank " << RANK << "]: " << x << endl;
/* #define DEBUG(x) */
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

void print(vector<double> &v) {
  for (auto &c : v) {
    cout << c << ",";
  }
  cout << endl;
}

void coord_print(double *c) {
  cout << "(";
  for (int i = 0; i < D; i++) {
    cout << c[i] << ",";
  }
  cout << ")";
}

void print(double *v, int n) {
  cout << "(rank " << RANK << ")";
  for (int i = 0; i < n; i++) {
    cout << v[i] << ",";
  }
  cout << endl;
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

inline double *coord_at(double *arr, int i) { return arr + i * D; }

inline void coord_copy(double *src, double *dst) {
  memcpy(dst, src, D * sizeof(double));
}

double coord_dist_square(double *p1, double *p2) {
  double dist = 0;
  for (int i = 0; i < D; i++) {
    dist += pow(p1[i] - p2[i], 2);
  }
  return dist;
}

int main(int argc, char *argv[]) {
  // init
  MPI::Init();
  RANK = COMM_WORLD.Get_rank();
  SIZE = COMM_WORLD.Get_size();

  MPI_Datatype MPI_D;
  MPI_Type_contiguous(D, MPI_DOUBLE, &MPI_D);
  MPI_Type_commit(&MPI_D);

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
    gen.rand_coord(positions + i * D);
  }

  // first centroid
  uniform_int_distribution<int> uni(0, n);
  if (RANK == MAIN) {
    coord_copy(positions + uni(gen.engine) * D, centroids);
  }
  COMM_WORLD.Bcast(centroids, D, MPI_DOUBLE, MAIN);

  // rest centroids
  discrete_distribution<int> rand_dx;
  double sum = 0;
  double *dmin = new double[n];
  double *round_centroids;
  double *round_sums;
  if (RANK == MAIN) {
    round_centroids = new double[SIZE * D];
    round_sums = new double[SIZE];
  }
  for (int r = 1; r < SIZE; ++r) {
    // 1.1 each choose a point
    // 1.1.1 each find min d(i) for all points, and total sum
    sum = 0;
    for (int i = 0; i < n; ++i) {
      dmin[i] = numeric_limits<double>::infinity();
      for (int j = 0; j < r; ++j) {
        double dist =
            coord_dist_square(coord_at(positions, i), coord_at(centroids, j));
        /* if (r == SIZE - 1) { */
        /*   DEBUG(STATUS("dist: " << dist)); */
        /* } */
        if (dist < dmin[i]) {
          /* if (r == SIZE - 1) { */
          /*   DEBUG(STATUS("dmin " << i << ": " << dist)); */
          /* } */
          dmin[i] = dist;
        }
      }
      sum += dmin[i];
    }
    // 1.1.2 each choose a point in proportion to dmin
    rand_dx = discrete_distribution<int>(dmin, dmin + N);
    coord_copy(coord_at(positions, rand_dx(gen.engine)),
               coord_at(centroids, r));

    DEBUG(STATUS("round: " << r << " sum: " << sum))
    // 1.1.3 gather all centroids and sums
    COMM_WORLD.Gather(&sum, 1, MPI_DOUBLE, round_sums, 1, MPI_DOUBLE, MAIN);

    COMM_WORLD.Gather(coord_at(centroids, r), D, MPI_DOUBLE, round_centroids, D,
                      MPI_DOUBLE, MAIN);

    /* // 1.1.4 main choose a centroid in proportion to sum and broadcast */
    if (RANK == MAIN) {
      rand_dx = discrete_distribution<int>(round_sums, round_sums + SIZE);
      coord_copy(coord_at(round_centroids, rand_dx(gen.engine)),
                 coord_at(centroids, r));
    }

    DEBUG(STATUS("round: " << r << " before bcast " << sum))
    /* COMM_WORLD.Bcast(coord_at(centroids, r), D, MPI_DOUBLE, MAIN); */
    DEBUG(STATUS("round: " << r << " done"))

    /* print(cluster.centroids[n_center]); */
  }

  MPI_Finalize();
  return 0;
}
