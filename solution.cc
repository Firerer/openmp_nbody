#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <openmpi/ompi/mpi/cxx/mpicxx.h>
#include <random>
#include <unistd.h>
#include <vector>

#define DEBUG(x) x
/* #define DEBUG2 */

#define ASSERT(x, msg)                                                         \
  if (!(x)) {                                                                  \
    cout << "ASSERTION FAILED: " << msg << endl;                               \
    exit(1);                                                                   \
  }
using namespace std;
using namespace MPI;
const double WEIGHT = 1;
const double INTERVAL = 1;
const int MASTER = 0;
int RANK;
int SIZE;
int N;
int D;

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

void crd_print(double *c) {
  printf("(");
  for (int i = 0; i < D; i++) {
    printf("%.2e", c[i]);
  }
  printf(")");
}

char *crd_cstr(char *str, double *c) {
  sprintf(str, "(");
  for (int i = 0; i < D; i++) {
    sprintf(str + strlen(str), "%.2e,", c[i]);
  }
  sprintf(str + strlen(str), ")");
  return str;
}

inline double *crd_at(double *arr, int i) { return arr + i * D; }

inline void crd_copy(double *src, double *dst) {
  /* for (int k = 0; k < D; k++) { */
  /*   dst[k] = src[k]; */
  /* } */
  memcpy(dst, src, D * sizeof(double));
}

double crd_dist_square(double *p1, double *p2) {
  double dist = 0;
  for (int i = 0; i < D; i++) {
    dist += pow(p1[i] - p2[i], 2);
  }
  return dist;
}

int crd_closest(double *p, double *centroids) {
  double min_dist = crd_dist_square(p, centroids);
  int min_idx = 0;
  double *c;
  double dist;
  for (int i = 1; i < SIZE; i++) {
    c = crd_at(centroids, i);
    dist = crd_dist_square(p, c);
    if (dist < min_dist) {
      min_dist = dist;
      min_idx = i;
    }
  }
  return min_idx;
}

void recalculate_centroid(double *c, double *points, int n) {
  memset(c, 0, D * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < D; j++) {
      c[j] += points[i * D + j];
    }
  }
  for (int j = 0; j < D; j++) {
    c[j] /= n;
  }
}

bool is_centroids_updated(double *old, double *ne) {
  for (int i = 0; i < SIZE * D; i++) {
    if (abs(old[i] - ne[i]) > 0.00001) {
      return 1;
    }
  }
  return 0;
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
  file.close();

  // init cluster
  int n = N / SIZE; // number of points
  if (RANK == MASTER) {
    n += N % SIZE;
  }
  double *centroids = (double *)malloc(SIZE * D * sizeof(double));
  // over allocate avoid realloc
  double *positions = (double *)malloc(N * D * sizeof(double));
  double *velocities = (double *)malloc(N * D * sizeof(double));
  double *psum = (double *)malloc(D * sizeof(double));
  for (int i = 0; i < N; i++) {
    gen.rand_coord(crd_at(positions, i));
  }

  // first centroid
  uniform_int_distribution<int> uni(0, n - 1);
  int first = uni(gen.engine);
  double *p = (double *)malloc(D * sizeof(double));
  COMM_WORLD.Allgather(crd_at(positions, first), D, MPI_DOUBLE, centroids, D,
                       MPI_DOUBLE);

  // kmeans
  //
  // prepare to sync points using alltoallv
  //
  char *str = (char *)malloc(sizeof(char) * 50);
  double *sendbuf = (double *)malloc(N * D * sizeof(double));
  int *target_centroids = (int *)malloc(N * sizeof(int));
  int *sendcounts = (int *)malloc(SIZE * sizeof(int));
  int *sdispls = (int *)malloc(SIZE * sizeof(int));
  int *recvcounts = (int *)malloc(SIZE * sizeof(int));
  int *rdispls = (int *)malloc(SIZE * sizeof(int));
  int r = 0;
  bool updated;

  do {
    printf("round %d\n", r);
#ifdef DEBUG2
    if (RANK == MASTER)
      printf("\n--- round %d ---\n", r);
    usleep(1000); // wait master finish print
    COMM_WORLD.Barrier();
#endif

    memset(sendcounts, 0, SIZE * sizeof(int));
    memset(rdispls, 0, SIZE * sizeof(int));

    // select centroids & calculate send counts
    for (int i = 0; i < n; i++) {
      int target = crd_closest(crd_at(positions, i), centroids);
      target_centroids[i] = target;
      sendcounts[target]++;
    }

    sdispls[0] = 0;
    for (int i = 1; i < SIZE; i++) {
      sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
    }

    // calculate offset in sendbuf for each point
    // and copy points to sendbuf
    for (int i = 0; i < n; i++) {
      int t = target_centroids[i];
      int offset = sdispls[t] + rdispls[t];
      crd_copy(crd_at(positions, i), crd_at(sendbuf, offset));
      rdispls[t]++;
#ifdef DEBUG3
      crd_cstr(str, crd_at(sendbuf, offset));
      printf(">>> %d -> %d: [%d] %s\n", RANK, t, i, str);
#endif
    }

#ifdef DEBUG3
    printf("> %d ", RANK);
    for (int i = 0; i < SIZE; i++) {
      printf(" [%d]num:%d,offset:%d, ", i, sendcounts[i], sdispls[i]);
    }
    printf("\n");
#endif

    // rdispls is nologer used
    // sync sendcounts
    COMM_WORLD.Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT);
    rdispls[0] = 0;
    for (int i = 1; i < SIZE; i++) {
      rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

#ifdef DEBUG3
    printf("<< %d ", RANK);
    for (int i = 0; i < SIZE; i++) {
      printf(" [%d] num: %d, offset: %d, ", i, recvcounts[i], rdispls[i]);
    }
    printf("\n");
#endif
    // update points count
    for (int i = 0; i < SIZE; i++) {
      n += recvcounts[i];
      n -= sendcounts[i];
    }
    // correct offsets
    for (int i = 0; i < SIZE; i++) {
      sendcounts[i] *= D;
      sdispls[i] *= D;
      recvcounts[i] *= D;
      rdispls[i] *= D;
    }
    COMM_WORLD.Alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, positions,
                         recvcounts, rdispls, MPI_DOUBLE);

    // update centroids
    // use velocities as temp
    memcpy(velocities, centroids, SIZE * D * sizeof(double));
    recalculate_centroid(p, positions, n);
    COMM_WORLD.Allgather(p, D, MPI_DOUBLE, centroids, D, MPI_DOUBLE);
    updated = is_centroids_updated(velocities, centroids);
#ifdef DEBUG2
    COMM_WORLD.Barrier();
    printf("\n!! Rank %d end round %d with %d points\n", RANK, r, n);
    printf("centroid: %s\n", crd_cstr(str, crd_at(centroids, RANK)));
    for (int i = 0; i < n; i++) {
      crd_cstr(str, crd_at(positions, i));
      printf("[%d]: %s\n", i, str);
    }
    COMM_WORLD.Barrier();
#endif
    r++;
  } while (updated);

  free(centroids);
  free(positions);
  free(velocities);
  free(psum);
  free(p);
  free(sendbuf);
  free(sendcounts);
  free(sdispls);
  free(recvcounts);
  free(rdispls);
  free(target_centroids);
  free(str);
  MPI_Finalize();
  return 0;
}
