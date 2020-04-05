#pragma once
#ifndef _USE_MATH_DEFINES
  #define _USE_MATH_DEFINES
#endif
#include <array>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

namespace nssc
{

enum Cell
{
  UNKNOWN = 0,
  DIR = 1,
  GHOST = 2
};

struct Discretization
{
  Discretization(size_t N)
      : resolution(N), h(1.0 / (resolution - 1)),
        C(4.0 / (h * h) + 4 * M_PI * M_PI), N(-1.0 / (h * h)),
        S(-1.0 / (h * h)), W(-1.0 / (h * h)), E(-1.0 / (h * h)) {}
  const int resolution;
  const double h, C, N, S, W, E;
};

double Solution(double x, double y)
{
  return sin(2 * M_PI * x) * sinh(2 * M_PI * y);
}

// 2D field with domain info and ghost layer
class Field
{
public:
  const Discretization disc;
  int N, DIM1, DIM2;

  std::vector<double> sol;  // solution
  std::vector<double> sol2; // solution swap
  std::vector<double> rhs;  // rhs
  std::vector<int> dom;     // domaininfo

  int mpi_rank;
  int mpi_numproc;

  std::string filename;
  std::ofstream out_file;

  ~Field(){ out_file.close();}
  Field(int N ,int rank, int numproc) : disc(N), N(N),mpi_rank(rank), mpi_numproc(numproc)
  {
    filename = "output_"+std::to_string(mpi_rank);
    try {
      out_file.open (filename, ios::out | ios::trunc);
    }
    catch (const ifstream::failure& e) {
      cout << "Exception opening/reading file";
      std::exit(1);
    }   
    std::cout << "Rank " << mpi_rank << std::endl;

    // allocate arrays
    int additionalLayer = 1; // always at least 1 neighbor ==> at least 1 additional ghost layer
    if ( mpi_rank == 0 || mpi_rank == mpi_numproc-1 ) // if not top or bot layer ==> 2 ghost layers
        additionalLayer = 2;
    int offset_j = mpi_rank*int(disc.resolution/mpi_numproc); // numproc == "number of domains"
    // Examples for offset calculation: -
    // 1) example with resolution = 64, numproc = 4
    // 64/4 = 16 ... should be offset

    // 2) example with resolution = 64, numproc = 3
    // 64/3 = 21.33333 ... should be offset
    // int(64/3) = 21 --> use that as offset. consequence: highest rank will get slightly bigger domain (by 1 layer?) --> should still be okay?
    // 3*21 = 63 --> 1 extra layer for highest rank

    // 3) example with resolution = 256, numproc = 3
    // 256/3 = 85.33333 ... should be offset
    // int(25/3) = 85 --> use that as offset. consequence: highest rank will get slightly bigger domain (by 1 layer?) --> should still be okay?
    // 3*85 = 255 --> 1 extra layer for highest rank  

    DIM1 = disc.resolution; // 1 dimension stays undecomped
    DIM2 = disc.resolution/mpi_numproc + additionalLayer; // how do we treat cases where (N/numproc) isn't integer?

    sol = std::vector<double>(DIM1 * DIM2, 0);
    sol2 = std::vector<double>(DIM1 * DIM2, 0);
    rhs = std::vector<double>(DIM1 * DIM2, 0);
    dom = std::vector<int>(DIM1 * DIM2, Cell::UNKNOWN);

    // setup domain, every point is innerDomain by default
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        // (top    || bot         || left   || right      )
        if (j == 0 || j == DIM2-1 || i == 0 || i == DIM1-1) // global domain boundary
          dom[i + DIM1 * j] = Cell::DIR;

        // (top && not top domain) || (bot         && not bot domain)
        if ( (j == 0 && mpi_rank != 0) || (j == DIM2-1 && mpi_rank != mpi_numproc-1)) // ghost layer
          dom[i + DIM1 * j] = Cell::GHOST;

      }
    }

    // init rhs
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN ||
            dom[i + DIM1 * j] == Cell::DIR)
        {
          rhs[i + DIM1 * j] =
              Solution(i * disc.h, (j+offset_j) * disc.h) * 4 * M_PI * M_PI;
        }
      }
    }

    // setup initial solution on global boundary
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::DIR)
        {
          sol[i + DIM1 * j] =
              Solution(i * disc.h, (j+offset_j) * disc.h);
          sol2[i + DIM1 * j] =
              Solution(i * disc.h, (j+offset_j) * disc.h);
        }
      }
    }
  }

  template <typename T>
  void printArray(std::vector<T> &v)
  {
    std::cout << std::defaultfloat;
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        std::cout << v[i + DIM1 * j]
                  << ",";
      }
      std::cout << std::endl;
    }
    std::cout << std::defaultfloat;
  }

  void printDomain(std::vector<Cell> &v)
  {
    std::vector<char> symbol_list { 'x', '#', 'o'};
    std::cout << std::defaultfloat;
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        std::cout << symbol_list[v[i + DIM1 * j]] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::defaultfloat;
  }

  // calculate residual
  void residual()
  {
    double max = 0;
    double sum = 0;
    int count = 0;
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN)
        {
          double tmp = rhs[i + DIM1 * j] -
                       (sol[(i + 0) + DIM1 * (j - 0)] * disc.C +
                        sol[(i + 1) + DIM1 * (j - 0)] * disc.E +
                        sol[(i - 1) + DIM1 * (j - 0)] * disc.W +
                        sol[(i + 0) + DIM1 * (j - 1)] * disc.S +
                        sol[(i + 0) + DIM1 * (j + 1)] * disc.N);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
          ++count;
        }
      }
    }

    double norm2 = sqrt(sum);
    double normMax = max;

    std::cout << std::scientific << "norm2res " << norm2 << std::endl;
    std::cout << std::scientific << "normMres " << normMax << std::endl;
  };
  void error()
  {
    double max = 0;
    double sum = 0;
    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN)
        {
          double tmp = sol[i + +DIM1 * j] -
                       Solution(i * disc.h, (j+2*mpi_rank) * disc.h);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
        }
      }
    }

    double norm2 = sqrt(sum);
    double normMax = max;

    std::cout << std::scientific << "norm2err " << norm2 << std::endl;
    std::cout << std::scientific << "normMerr " << normMax << std::endl;
  };

  // perform Jacobi Iteration, with optional skip range
  void solve(int iterations)
  {

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    double runtime;

    start = std::chrono::high_resolution_clock::now();

    int iter;
    for (iter = 1; iter <= iterations; ++iter)
    {
      update();
      // Synchronize ???
    }

    end = std::chrono::high_resolution_clock::now();
    runtime =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();

    std::cout << std::scientific << "runtime " << runtime << std::endl;
    std::cout << std::scientific << "runtime/iter " << runtime / iter << std::endl;
  }

  void update()
  {
    for (int j = 1; j < DIM2 - 1; ++j)
    {
      for (int i = 1; i < DIM1 - 1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN)
        {
          sol2[i + DIM1 * j] =
              1.0 / disc.C *
              (rhs[i + DIM1 * j] -
               (sol[(i + 1) + DIM1 * (j - 0)] * disc.E +
                sol[(i - 1) + DIM1 * (j - 0)] * disc.W +
                sol[(i + 0) + DIM1 * (j - 1)] * disc.S +
                sol[(i + 0) + DIM1 * (j + 1)] * disc.N));
        }
      }
    }
    sol.swap(sol2);
  };
};

} // namespace nssc
