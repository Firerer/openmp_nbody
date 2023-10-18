#include <float.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <openmpi/ompi/mpi/cxx/mpicxx.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* #define PLOT */
#define kmeanspp
#define TIMING
// #define FIX_R

using namespace std;
using namespace MPI;
#define WEIGHT 1.0
#define INTERVAL 0.1 / 2
#define G 1
#define MASTER 0
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

  void rand_coord(double *p) {
    int i = dd(engine);
    for (int j = 0; j < D; j++) {
      p[j] = nd(engine) * sds[i] + means[i][j];
    }
  }
};

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
  memcpy(dst, src, D * sizeof(double));
}

double crd_dist_square(double *p1, double *p2) {
  double dist = 0;
  for (int i = 0; i < D; i++) {
    dist += pow(p1[i] - p2[i], 2);
  }
  return dist;
}

double crd_dist(double *p1, double *p2) {
  double dist = 0;
  for (int i = 0; i < D; i++) {
    dist += abs(p1[i] - p2[i]);
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

void cluster_centroid(double *centroid, double *positions, int n) {
  memset(centroid, 0, D * sizeof(double));
  for (int j = 0; j < D; j++) {
#pragma omp parallel for reduction(+ : centroid[j])
    for (int i = 0; i < n; i++) {
      centroid[j] += positions[i * D + j];
    }
  }
  for (int j = 0; j < D; j++) {
    centroid[j] /= n;
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

double cluster_variance(double *centroids, double *positions, int n) {
  if (n < 2) {
    return 0.0;
  }
  double variance = 0;
  double avgs[D]; // for centroids
  memset(avgs, 0, D * sizeof(double));

  for (int j = 0; j < D; j++) {
#pragma omp parallel for reduction(+ : avgs[j])
    for (int i = 0; i < n; i++) {
      avgs[j] += positions[i * D + j];
    }
  }
  for (int j = 0; j < D; j++) {
    avgs[j] /= n;
  }
  for (int j = 0; j < D; j++) {
#pragma omp parallel for reduction(+ : variance)
    for (int i = 0; i < n; i++) {
      variance += pow(positions[i * D + j] - avgs[j], 2);
    }
  }
  variance /= n;
  return variance;
}

#define MAX_FORCE 1.0
void cal_force(double *force, int at, double *positions, int *ns,
               double *centroids) {
  double *target = crd_at(positions, at);
  memset(force, 0, D * sizeof(double));
  double dist;
  double *pi;
  // force from other clusters
  for (int i = 0; i < SIZE; i++) {
    if (i == RANK) {
      continue;
    }
    pi = crd_at(centroids, i);
    dist = crd_dist_square(target, pi);
    if (dist == 0 || dist > 100) {
      continue;
    }
    for (int j = 0; j < D; j++) {
      double added =
          ns[i] * WEIGHT * G * (pi[j] - target[j]) / pow(dist, 3 / 2);
      added = added > 0 ? min(added, MAX_FORCE) : max(added, -MAX_FORCE);
      force[j] += added;
    }
  }

/* int hole_mass = N / SIZE; */
/* // force from blackhole at center */
/* for (int j = 0; j < D; j++) { */
/*   double added = N * G * (0 - target[j]) / pow(abs(target[j]), 3 / 2); */
/*   added = added > 0 ? min(added, MAX_FORCE) : max(added, -MAX_FORCE); */
/*   force[j] += added; */
/* } */

// force from other points
#pragma omp parallel for private(pi, dist)
  for (int i = 0; i < ns[RANK]; i++) {
    if (at == i) {
      continue;
    }
    pi = crd_at(positions, i);
    dist = crd_dist_square(target, pi);
    for (int j = 0; j < D; j++) {
      double added =
          WEIGHT * WEIGHT * G * (pi[j] - target[j]) / pow(dist, 3 / 2);
      added = added > 0 ? min(added, MAX_FORCE) : max(added, -MAX_FORCE);
#pragma omp atomic
      force[j] += added;
    }
  }
}

// return new n for current rank
int kmeans(double *centroids, double *positions, int n, int &r) {
  double *sendbuf = (double *)malloc(N * D * sizeof(double));
  int *target_centroids = (int *)malloc(N * sizeof(int));
  int sendcounts[SIZE];
  int sdispls[SIZE];
  int recvcounts[SIZE];
  int rdispls[SIZE];
  r = 0;
  /* int r = 0; */
  /* int *r= round; */
  double *old_centroids = (double *)malloc(N * D * sizeof(double));
  memcpy(old_centroids, centroids, SIZE * D * sizeof(double));
  double tmp[D];

  do {
    memset(sendcounts, 0, SIZE * sizeof(int));
    memset(rdispls, 0, SIZE * sizeof(int));

// select centroids & calculate send counts
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      int target = crd_closest(crd_at(positions, i), centroids);
      target_centroids[i] = target;
#pragma omp atomic
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
    }

    // rdispls is nologer used
    // sync sendcounts
    COMM_WORLD.Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT);
    rdispls[0] = 0;
    for (int i = 1; i < SIZE; i++) {
      rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

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
    memcpy(old_centroids, centroids, SIZE * D * sizeof(double));
    for (int i = 0; i < SIZE; i++) {
      cluster_centroid(tmp, positions, n);
    }
    COMM_WORLD.Allgather(tmp, D, MPI_DOUBLE, centroids, D, MPI_DOUBLE);

    r++;
  } while (is_centroids_updated(old_centroids, centroids));

  free(old_centroids);
  free(sendbuf);
  free(target_centroids);
  return n;
}

int main(int argc, char *argv[]) {
#ifdef TIMING
  clock_t start_t = clock();
#endif
  // init
  MPI::Init();
  RANK = COMM_WORLD.Get_rank();
  SIZE = COMM_WORLD.Get_size();
#ifdef PLOT
  char filename[20];
  snprintf(filename, sizeof(filename), "log/thread_%d.log", RANK);
  FILE *log = fopen(filename, "w");
#endif

  // all read file avoid communication
  ifstream file = ifstream(argv[1]);
  file >> N >> D;
  // init cluster
  int n = N / SIZE; // number of points
  if (RANK == MASTER) {
    n += N % SIZE;
  }
#ifdef TIMING
  clock_t end_init_t = clock();
#endif

  // over allocate avoid realloc
  double *positions = (double *)malloc(N * D * sizeof(double));

  Generator gen(RANK, file);
  file.close();
  for (int i = 0; i < n; i++) {
    gen.rand_coord(crd_at(positions, i));
  }

#ifdef TIMING
  clock_t end_gen_t = clock();
#endif

  double centroids[SIZE * D];
  // first centroid
  COMM_WORLD.Allgather(crd_at(positions, 0), D, MPI_DOUBLE, centroids, D,
                       MPI_DOUBLE);
#ifdef kmeanspp
  /* COMM_WORLD.Bcast(crd_at(positions, 0), D, MPI_DOUBLE, MASTER); */
  /* crd_copy(positions, centroids); */
  /* char str[50]; */
  /* printf("RANK %d first centroids %s\n", RANK, crd_cstr(str, centroids)); */
  // rest centroids
  discrete_distribution<int> rand_dx;
  double sum = 0;
  double *dmin = new double[n];
  double round_centroids[SIZE * D];
  double round_sums[SIZE];
  for (int r = 1; r < SIZE; ++r) {
    // 1.1 each choose a point
    // 1.1.1 each find min d(i) for all points, and total sum
    sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; ++i) {
      dmin[i] = numeric_limits<double>::infinity();
      for (int j = 0; j < r; ++j) {
        double dist =
            crd_dist_square(crd_at(positions, i), crd_at(centroids, j));
        if (dist < dmin[i]) {
          dmin[i] = dist;
        }
      }
      sum += dmin[i];
    }
    // 1.1.2 each choose a point in proportion to dmin
    int randn = discrete_distribution<int>(dmin, dmin + n)(gen.engine);
    crd_copy(crd_at(positions, randn), crd_at(centroids, r));

    // 1.1.3 gather all centroids and sums
    COMM_WORLD.Gather(&sum, 1, MPI_DOUBLE, round_sums, 1, MPI_DOUBLE, MASTER);

    COMM_WORLD.Gather(crd_at(centroids, r), D, MPI_DOUBLE, round_centroids, D,
                      MPI_DOUBLE, MASTER);

    /* // 1.1.4 MASTER choose a centroid in proportion to sum and broadcast */
    if (RANK == MASTER) {
      rand_dx = discrete_distribution<int>(round_sums, round_sums + SIZE);
      crd_copy(crd_at(round_centroids, rand_dx(gen.engine)),
               crd_at(centroids, r));
    }

    COMM_WORLD.Bcast(crd_at(centroids, r), D, MPI_DOUBLE, MASTER);
  }
#endif
  int kemans_r = 0;
  n = kmeans(centroids, positions, n, kemans_r);

#ifdef TIMING
  clock_t end_kmeans_t = clock();
#endif
  ///
  /// simulation
  ///
  // sync n to all ranks
  int ns[SIZE];
  COMM_WORLD.Allgather(&n, 1, MPI_INT, ns, 1, MPI_INT);
  double force[D];
  double *velocities = (double *)malloc(N * D * sizeof(double));
  memset(velocities, 0, N * D * sizeof(double));
  int r = 0;
  double variance = cluster_variance(centroids, positions, n);
  double sumv;
  COMM_WORLD.Allreduce(&variance, &sumv, 1, MPI_DOUBLE, MPI_SUM);
  double init_variance = sumv;
#ifdef PLOT
  char ps[50];
  int k;
  for (int i = 0; i < n; i++) {
    k = 0;
    k += sprintf(ps + k, "||%d,%d,", RANK, r);
    for (int j = 0; j < D; j++) {
      k += sprintf(ps + k, "%.2e,", positions[i * D + j]);
    }
    k += sprintf(ps + k, "%d,", i);
    fprintf(log, "%s\n", ps);
  }
#endif
  while (r < 150 && sumv > init_variance / 4) {
#pragma omp parallel for private(force)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < D; j++) {
        cal_force(force, i, positions, ns, centroids);
        positions[i * D + j] += velocities[i * D + j] * INTERVAL;
        velocities[i * D + j] += force[j] * INTERVAL / WEIGHT;
      }
    }

#ifdef PLOT
    for (int i = 0; i < n; i++) {
      k = 0;
      k += sprintf(ps + k, "||%d,%d,", RANK, r);
      for (int j = 0; j < D; j++) {
        k += sprintf(ps + k, "%.2e,", positions[i * D + j]);
      }
      k += sprintf(ps + k, "%d,", i);
      fprintf(log, "%s\n", ps);
    }
#endif // DEBUG
    // prepare for next round
    variance = cluster_variance(centroids, positions, n);
    COMM_WORLD.Allreduce(&variance, &sumv, 1, MPI_DOUBLE, MPI_SUM);
    /* printf("RANK %d round %d variance %lf\n", RANK, r, sumv); */
    r++;
  }

#ifdef TIMING
  clock_t end_t = clock();
#endif
  free(velocities);
  free(positions);

#ifdef TIMING
  if (RANK == MASTER) {
    const char *csvname = "record.csv";
    FILE *csv;
    if (access(csvname, F_OK) != 0) {
      csv = fopen(csvname, "w");
      fprintf(csv, "N,k,D,c,kr,r,total,gen,kmeans,simulation\n");
    } else {
      csv = fopen(csvname, "a");
    }
    fprintf(csv, "%d,%d,%d,%ld,%d,%d,%lf,%lf,%lf,%lf\n", N, SIZE, D,
            gen.means.size(), kemans_r, r,
            (double)(clock() - start_t) / CLOCKS_PER_SEC,
            (double)(end_gen_t - end_init_t) / CLOCKS_PER_SEC,
            (double)(end_kmeans_t - end_gen_t) / CLOCKS_PER_SEC,
            (double)(end_t - end_kmeans_t) / CLOCKS_PER_SEC);
  }
#endif
  MPI_Finalize();
  return 0;
}
