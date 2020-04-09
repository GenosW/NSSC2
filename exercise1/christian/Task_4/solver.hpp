// #pragma once
#include <array>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>
#include <fstream>

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
  MPI_Status stat;
  const Discretization disc;
  int resolution, DIM1, DIM2, M, N, m, n;

  std::vector<double> sol;  // solution
  std::vector<double> sol2; // solution swap
  std::vector<double> rhs;  // rhs
  std::vector<int> dom;     // domaininfo

  int mpi_rank;
  int mpi_numproc;

  ~Field(){ }
  Field(int resolution ,int rank, int numproc) : disc(resolution), resolution(resolution), mpi_rank(rank), mpi_numproc(numproc)
  {


    std::cout << "Rank " << mpi_rank << std::endl;
    string name = "rank" + std::to_string(mpi_rank) + ".txt";

    // allocate arrays
    M = 1;
    N = 1;
    factorization(mpi_numproc);
    std::cout << M << N << endl;
    m = mpi_rank%M;
    n = (mpi_rank-m)/M;

    int additionalLayer_X = 2;
    int additionalLayer_Y = 2;
    if ( m == 0 || m == M-1 )
        additionalLayer_X = 1;
    if ( M == 1 )
        additionalLayer_X = 0;
    if ( n == 0 || n == N-1 )
        additionalLayer_Y = 1;
    if (mpi_numproc == 1)
        additionalLayer_Y = 0;

    if ( m <= M-2 )
        DIM1 = (int)std::floor((double)resolution/M) + additionalLayer_X;
    else
        DIM1 = resolution - (int)std::floor((double)resolution/M)*(M-1) + additionalLayer_X;
    if ( n <= N-2 )
        DIM2 = (int)std::floor((double)resolution/N) + additionalLayer_Y;
    else
        DIM2 = resolution - (int)std::floor((double)resolution/N)*(N-1) + additionalLayer_Y;

    sol = std::vector<double>(DIM1 * DIM2, 0);
    sol2 = std::vector<double>(DIM1 * DIM2, 0);
    rhs = std::vector<double>(DIM1 * DIM2, 0);
    dom = std::vector<int>(DIM1 * DIM2, Cell::UNKNOWN);

///////////////////////////////////////////////////////////////////////////////////////////// setup local domain

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if ( i==0 || i==DIM1-1 || j==0 || j==DIM2-1 ) // ghost layer
          dom[i + DIM1 * j] = Cell::GHOST;

        if ( (m==0 && i==0) || (m==M-1 && i==DIM1-1) || (n==0 && j==0) || (n==N-1 && j==DIM2-1)) // global domain boundary
          dom[i + DIM1 * j] = Cell::DIR;
      }
    }
    printLocalDomain(name);

////////////////////////////////////////////////////////////////////////////////////////////////// init local rhs

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN || dom[i + DIM1 * j] == Cell::DIR)
        {
          rhs[i + DIM1 * j] =
              Solution(real_x(i) * disc.h, real_y(j) * disc.h) * 4 * M_PI * M_PI;
        }
      }
    }

///////////////////////////////////////////////////////////////////////////////////////////////////// setup initial solution on local boundary

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::DIR)
        {
          sol[i + DIM1 * j] =
              Solution(real_x(i) * disc.h, real_y(j) * disc.h);
          sol2[i + DIM1 * j] =
              Solution(real_x(i) * disc.h, real_y(j) * disc.h);
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
          int ig = i - 1;
          int jg = j - 1;
          double tmp = sol[i + +DIM1 * j] -
                       Solution(ig * disc.h, jg * disc.h);

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

/////////////////////////////////////////////////////////////////////////////////////////////// perform Jacobi Iteration, with optional skip range
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

    ///////////////////////// vertical communication /////////////////////////////////

    if (n > 0)
    {
        double msg_upper[DIM1];
        for (int i = 0; i < DIM1; ++i)
            msg_upper[i] = sol[DIM1+i];
        
        MPI_Send(msg_upper, DIM1, MPI_DOUBLE, (n-1)*M+m, 1, MPI_COMM_WORLD);
    }
    if (n < N-1)
    {
        double rec_upper[DIM1];
        MPI_Recv(rec_upper, DIM1, MPI_DOUBLE, (n+1)*M+m, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < DIM1; ++i)
            sol[(DIM2-1)*DIM1+i] = rec_upper[i];
    }

    if (n < N-1)
    {
        double msg_lower[DIM1];
        for (int i = 0; i < DIM1; ++i)
            msg_lower[i] = sol[DIM1*(DIM2-2)+i];
        
        MPI_Send(msg_lower, DIM1, MPI_DOUBLE, (n+1)*M+m, 1, MPI_COMM_WORLD);
    }
    if (n > 0)
    {
        double rec_lower[DIM1];
        MPI_Recv(rec_lower, DIM1, MPI_DOUBLE, (n-1)*M+m, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < DIM1; ++i)
            sol[i] = rec_lower[i];
    }

///////////////////////// horizontal communication /////////////////////////////////

    if (m > 0)
    {
        double msg_left[DIM2];
        for (int j = 0; j < DIM2; ++j)
            msg_left[j] = sol[DIM1*j];
        
        MPI_Send(msg_left, DIM2, MPI_DOUBLE, n*M+m-1, 1, MPI_COMM_WORLD);
    }
    if (m < M-1)
    {
        double rec_left[DIM2];
        MPI_Recv(rec_left, DIM2, MPI_DOUBLE, n*M+m+1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int j = 0; j < DIM2; ++j)
            sol[(DIM1-2)+j*DIM1] = rec_left[j];
    }

    if (m < M-1)
    {
        double msg_right[DIM2];
        for (int j = 0; j < DIM2; ++j)
            msg_right[j] = sol[(DIM1-1)+j*DIM1];
        
        MPI_Send(msg_right, DIM2, MPI_DOUBLE, n*M+m+1, 1, MPI_COMM_WORLD);
    }
    if (m > 0)
    {
        double rec_right[DIM2];
        MPI_Recv(rec_right, DIM2, MPI_DOUBLE, n*M+m-1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int j = 0; j < DIM2; ++j)
            sol[j*DIM1] = rec_right[j];
    }

  };

///////////////////////////////////////////////////////////////////////////////////////////////////////////// factors for domain decomposition

    void factorization(int R)
      {
        int Rc = R;
        double c = 0;
        int f[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,97,101};

        for (int i = 0; i < 25; i++)
        {
            while (R%f[i] == 0)
            {
                R /= f[i];
                c++;
            }
        }
        
        int factors[int(c)];
        c = 0;
            
        for (int i = 0; i < 13; i++)
        {
            while (Rc%f[i] == 0)
            {
                Rc /= f[i];
                factors[int(c)] = f[i];
                c++;
            }
        }
            if (int(c)%2 == 1)
            {
                for (int j = 0; j < std::ceil(c/2.0); j++)
                {
                    N *= factors[j];
                }
                for (int j = std::ceil(c/2.0); j < c; j++)
                {
                    M *= factors[j];
                }
            }
            else
            {
                for (int j = 0; j < c; j++)
                {
                    if (j%2 == 0)
                        N *= factors[j];
                    else
                        M *= factors[j];
                }
            }
      };

////////////////////////////////////////////////////////////////////////////////////////////// print local Domain

void printLocalDomain( string name)
      {
        ofstream outfile;
        outfile.open(name, ios::out | ios::trunc);
        outfile << "Rank " << mpi_rank << ", m: " << m << ", n: " << n << endl << endl;
        outfile << "\t";
        std::vector<char> symbol_list { 'x', '#', 'o'};
        outfile << std::defaultfloat;
        for (int i = 0; i < DIM1; i++)
            outfile << real_x(i) << "\t";
        outfile << endl;
        for (int j = 0; j < DIM2; ++j)
        {
            outfile << real_y(j) << "\t";
          for (int i = 0; i < DIM1; ++i)
          {
            outfile << symbol_list[dom[i + DIM1 * j]] << "\t";
          }
          outfile << std::endl;
        }
        outfile << std::defaultfloat;
        outfile.close();   
      };


////////////////////////////////////////////////////////////////////////////////////////////////// real X and Y

int real_x(int i)
    {
        int x = i + m*(int)std::floor((double)resolution/M) - min(m,1);
        return x;
    };

int real_y(int j)
    {
        int y = j + n*(int)std::floor((double)resolution/N) - min(n,1);
        return y;
    };


};


} // namespace nssc
