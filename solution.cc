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
const int G = 1;
const int MAIN = 0;
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

void print(double *v, int n) {
  cout << "(rank " << RANK << ")";
  for (int i = 0; i < n; i++) {
    cout << v[i] << ",";
  }
  cout << endl;
}

struct Coord {
  double *coord;
  Coord() { coord = new double[D]; }
  Coord(double *coord) { this->coord = coord; }
  ~Coord() { delete[] coord; }
  double &operator[](int i) {
    DEBUG(ASSERT(0 <= i && i < D, "Coord[] out of range");)
    return coord[i];
  }
  double *begin() { return coord; }
  double *end() { return coord + D; }
};

struct Coords {
  double *coords;
  int n;
  Coords(int n) {
    coords = new double[n * D];
    this->n = n;
  }
  ~Coords() { delete[] coords; }
  Coord operator[](int i) {
    DEBUG(ASSERT(0 <= i && i < D, "Coords[] out of range");)
    return Coord(coords + i * D);
  }
  double *begin() { return coords; }
  double *end() { return coords + n * D; }
};

struct Point {
  vector<double> pos;
  vector<double> velocity;
  Point() {
    pos = vector<double>(D);
    velocity = vector<double>(D);
  }

  // TODO
  double distance(vector<double> &p) {
    double dist = 0;
    for (int i = 0; i < D; i++) {
      dist += pow((this->pos[i] - p[i]), 2);
    }
    return dist;
  }

  vector<double> vec_square(vector<double> &p) {
    vector<double> dist(D);
    for (int i = 0; i < D; i++) {
      dist[i] = ldexp(pos[i] - p[i], 2);
    }
    return dist;
  }

  double dist_square(vector<double> &p) {
    double dist = 0;
    for (int i = 0; i < D; i++) {
      dist += pow(pos[i] - p[i], 2);
    }
    return dist;
  }

  void print() {
    cout << "Point(";
    for (const auto &c : pos) {
      cout << c << ",";
    }
    cout << ")";
  }

  /* private: */
  /*   // prevent copy */
  /*   Point &operator=(const Point &); */
};

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
  double gaussian(double &mean, double &sd) { return normal() * sd + mean; }
  void point_init(Point &p) {
    int i = dd(engine);
    for (int j = 0; j < D; j++) {
      p.pos[j] = nd(engine) * sds[i] + means[i][j];
    }
  }
};

void points_distance() {}

bool is_close(Point &p1, Point &p2) { return false; }

struct Cluster {
  vector<double> psum;
  vector<Point> points;
  vector<vector<double>> centroids;
  double **centroids_;
  double *centroids_data;

  Cluster(Generator gen) {
    psum = vector<double>(D);
    centroids_data = new double[N * D];
    centroids = vector<vector<double>>(N, vector<double>(D));
    points = vector<Point>(N);
    for (int i = 0; i < N; i++) {
      gen.point_init(points[i]);

      for (int j = 0; j < D; j++) {
        psum[j] += points[i].pos[j];
      }
    }
  }

  void add(Point &p) {
    for (int i = 0; i < D; i++) {
      psum[i] += p.pos[i];
    }
    points.push_back(p);
  }

  Point remove(int p) {
    for (int i = 0; i < D; i++) {
      psum[i] -= points[p].pos[i];
    }
    Point tmp = points[p];
    points[p] = points.back();
    points.pop_back();
    return tmp;
  }

  void print() {
    for (auto &p : points) {
      p.print();
      cout << endl;
    }
  }

  void center_mass() {
    /* double mass = points.size(); */
    /* vector<double> force(D); */
  }

private:
  // prevent copy
  Cluster &operator=(const Cluster &);
};

int main(int argc, char *argv[]) {
  // init
  MPI::Init();
  RANK = COMM_WORLD.Get_rank();
  SIZE = COMM_WORLD.Get_size();
  cout << "Hello from " << RANK << " of " << SIZE << endl;

  MPI_Datatype MPI_D;
  MPI_Type_contiguous(D, MPI_DOUBLE, &MPI_D);
  MPI_Type_commit(&MPI_D);

  // all read file avoid communication
  ifstream file = ifstream(argv[1]);
  file >> N >> D;
  Generator gen(RANK, file);
  Cluster cluster(gen);

  // first centroid
  uniform_int_distribution<int> uni(0, N);
  int first = uni(gen.engine);
  if (RANK == MAIN) {
    cluster.centroids[0] = cluster.points[first].pos;
  }
  COMM_WORLD.Bcast(cluster.centroids[0].data(), D, MPI_DOUBLE, MAIN);

  // rest centroids
  int n_center = 1;
  discrete_distribution<int> rand_dx;
  vector<double> dmin(N);

  // 1.1 each choose a point

  // 1.1.1 each find min d(i) for all points, and total sum
  double sum = 0;
  for (int i = 0; i < N; ++i) {
    dmin[i] = numeric_limits<double>::infinity();
    for (int j = 0; j < n_center; ++j) {
      double dist = cluster.points[i].dist_square(cluster.centroids[j]);
      if (dist < dmin[i]) {
        dmin[i] = dist;
      }
    }
    sum += dmin[i];
  }
  // 1.1.2 each choose a point in proportion to dmin
  rand_dx = discrete_distribution<int>(dmin.begin(), dmin.end());
  int r = rand_dx(gen.engine);
  double *centroid = cluster.points[r].pos.data();
  vector<double> centroids;
  Coord centroid_(cluster.points[r].pos.data());
  Coords centroids_;
  vector<double> sums;

  if (RANK == MAIN) {
    centroids = vector<double>(SIZE * D);
    sums = vector<double>(SIZE);
    centroids_ = Coords(SIZE);
  }
  // 1.1.3 gather all centroids and sums
  COMM_WORLD.Gather(centroid, D, MPI_DOUBLE, centroids_.begin(), D, MPI_DOUBLE,
                    MAIN);
  COMM_WORLD.Gather(&sum, 1, MPI_DOUBLE, sums.data(), 1, MPI_DOUBLE, MAIN);
  // 1.1.4 main choose a centroid in proportion to sum and broadcast
  int cluster_i;
  if (RANK == MAIN) {
    rand_dx = discrete_distribution<int>(dmin.begin(), dmin.end());
    cluster_i = rand_dx(gen.engine);
    centroid = centroids.data() + cluster_i * D;
  }
  COMM_WORLD.Bcast(centroid, D, MPI_DOUBLE, MAIN);

  // 1.1.5 each assign new centroid
  /* cluster.centroids[n_center] = centroid; */
  n_center += 1;

  Coords coords(10);
  print(centroid, D);

  MPI_Finalize();
  return 0;
}
