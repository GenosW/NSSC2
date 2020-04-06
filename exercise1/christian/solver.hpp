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
#include <string>
#include <fstream>

namespace nssc
{

/////////////////////////////////// Definition of Cell //////////////////////////////////////////

enum Cell
{
  UNKNOWN = 0,
  DIR = 1,
  GHOST = 2
};

/////////////////////////////////// Definition of Discretization ///////////////////////////////

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

//////////////////////////////////////////////// 2D field with domain info and ghost layer //////////////////////////////////////////////////////
class Field
{
public:
  MPI_Status stat;
  const Discretization disc;
  int N, DIM1, DIM2;

  std::vector<double> sol;  // solution
  std::vector<double> sol2; // solution swap
  std::vector<double> rhs;  // rhs
  std::vector<int> dom;     // domaininfo
  std::vector<double> sol_global;
  std::vector<int> dom_global;
  std::vector<double> rhs_global;

  int mpi_rank;
  int mpi_numproc;

/////////////////////////////////////////////////////////// Destructor ///////////////////////////////////////////////////////

  ~Field(){ }

/////////////////////////////////////////////////////////// Konstruktor ////////////////////////////////////////////////////////

  Field(int N ,int rank, int numproc) : disc(N), N(N),mpi_rank(rank), mpi_numproc(numproc)
  {

    string name = "rank" + std::to_string(mpi_rank) + ".txt";
    std::cout << "Rank " << mpi_rank << std::endl;

    // allocate arrays
    int additionalLayer = 2;
    if ( mpi_rank == 0 || mpi_rank == mpi_numproc-1 )
        additionalLayer = 1;
    if (mpi_numproc == 1)
        additionalLayer = 0;

    DIM1 = N;
    DIM2 = N/mpi_numproc + additionalLayer;
    

    sol = std::vector<double>(DIM1 * DIM2, 0);
    sol2 = std::vector<double>(DIM1 * DIM2, 0);
    sol_global = std::vector<double>(N*N, 0);
    rhs = std::vector<double>(DIM1 * DIM2, 0);
    rhs_global = std::vector<double>(N*N, 0);
    dom = std::vector<int>(DIM1 * DIM2, Cell::UNKNOWN);
    dom_global = std::vector<int>(N*N, Cell::UNKNOWN);

////////////////////////////////////////////////////////////////////////// setup domain, every point is innerDomain by default

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {

        if (j == 0 || j == DIM2-1 || i == 0 || i == DIM1-1) // global domain boundary
          dom[i + DIM1 * j] = Cell::DIR;

        if ( (j == 0 && mpi_rank != 0) || (j == DIM2-1 && mpi_rank != mpi_numproc-1)) // ghost layer
          dom[i + DIM1 * j] = Cell::GHOST;

      }
    }
    printDomain(dom,name);

////////////////////////////////////////////////////////////////////////////// init rhs

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::UNKNOWN ||
            dom[i + DIM1 * j] == Cell::DIR)
        {
          rhs[i + DIM1 * j] =
              Solution(i * disc.h, (j+mpi_rank*N/mpi_numproc-std::min(mpi_rank,1)) * disc.h) * 4 * M_PI * M_PI;
        }
      }
    }

////////////////////////////////////////////////////////////////////////////////// setup initial solution on local boundary

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if (dom[i + DIM1 * j] == Cell::DIR)
        {
          sol[i + DIM1 * j] =
              Solution(i * disc.h, (j+mpi_rank*N/mpi_numproc-std::min(mpi_rank,1)) * disc.h);
          sol2[i + DIM1 * j] =
              Solution(i * disc.h, (j+mpi_rank*N/mpi_numproc-std::min(mpi_rank,1)) * disc.h);
        }
      }
    }
    string namesolution = "rank" + std::to_string(mpi_rank) + "_initialSolution.txt";
    printSolution(sol, namesolution);
  }

/////////////////////////////////////////////////////////////////////////////////// printArray

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

/////////////////////////////////////////////////////////////////////////////////// calculate residual

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


/////////////////////////////////////////////////////////////////////////////////////// calculate residual global

  void residual_global()
  {
    if ( mpi_rank == 0)
    {
    double max = 0;
    double sum = 0;
    int count = 0;
    for (int j = 0; j != N; ++j)
    {
      for (int i = 0; i != N; ++i)
      {
        if (dom_global[i + N * j] == Cell::UNKNOWN)
        {
          double tmp = Solution(i * disc.h, j * disc.h) * 4 * M_PI * M_PI -
                       (sol_global[(i + 0) + N * (j - 0)] * disc.C +
                        sol_global[(i + 1) + N * (j - 0)] * disc.E +
                        sol_global[(i - 1) + N * (j - 0)] * disc.W +
                        sol_global[(i + 0) + N * (j - 1)] * disc.S +
                        sol_global[(i + 0) + N * (j + 1)] * disc.N);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
          ++count;
        }
      }
    }

    double norm2 = sqrt(sum);
    double normMax = max;

    std::cout << std::scientific << "norm2res_gloabl " << norm2 << std::endl;
    std::cout << std::scientific << "normMres_global " << normMax << std::endl;
    }
  };

////////////////////////////////////////////////////////////////////////////////////////////// calculate error

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
          double tmp = sol[i +DIM1 * j] -
                       Solution(i * disc.h, (j+mpi_rank*N/mpi_numproc-std::min(mpi_rank,1)) * disc.h);

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

//////////////////////////////////////////////////////////////////////////////////////////////////////// calculate error global

void error_global()
  {
    if (mpi_rank == 0)
    {
    double max = 0;
    double sum = 0;
    for (int j = 0; j != N; ++j)
    {
      for (int i = 0; i != N; ++i)
      {
        if (dom_global[i + N * j] == Cell::UNKNOWN)
        {
          double tmp = sol_global[i +N * j] -
                       Solution(i * disc.h, j * disc.h);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
        }
      }
    }

    double norm2 = sqrt(sum);
    double normMax = max;

    std::cout << std::scientific << "norm2err_global " << norm2 << std::endl;
    std::cout << std::scientific << "normMerr_global " << normMax << std::endl;

    }
  };



/////////////////////////////////////////////////////////////////////////////////////////////////////// perform Jacobi Iteration, with optional skip range

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

        assemble_Original_Domain_and_Solution();
        if ( mpi_rank == 0)
        {
            printDomain(dom_global,"GlobalDomain");
            printSolution(sol_global, "GlobalSolution");
        }
        

        end = std::chrono::high_resolution_clock::now();
        runtime =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
                .count();

        //std::cout << std::scientific << "iterations " << iterations << std::endl;
        //std::cout << std::scientific << "runtime " << runtime << std::endl;
        //std::cout << std::scientific << "runtime/iter " << runtime / iter << std::endl;
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////// update function 

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
    
    
    if (mpi_rank > 0)
    {
        double msg_upper[DIM1];
        for (int i = 0; i < DIM1; ++i)
            msg_upper[i] = sol[DIM1+i];
        
        MPI_Send(msg_upper, DIM1, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD);
    }
    if (mpi_rank < mpi_numproc-1)
    {
        double rec_upper[DIM1];
        MPI_Recv(rec_upper, DIM1, MPI_DOUBLE, mpi_rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < DIM1; ++i)
            sol[(DIM2-1)*DIM1+i] = rec_upper[i];
    }

    if (mpi_rank < mpi_numproc-1)
    {
        double msg_lower[DIM1];
        for (int i = 0; i < DIM1; ++i)
            msg_lower[i] = sol[DIM1*(DIM2-2)+i];
        
        MPI_Send(msg_lower, DIM1, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD);
    }
    if (mpi_rank > 0)
    {
        double rec_lower[DIM1];
        MPI_Recv(rec_lower, DIM1, MPI_DOUBLE, mpi_rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < DIM1; ++i)
            sol[i] = rec_lower[i];
    }
    string namesolution = "rank" + std::to_string(mpi_rank) + "_zwischenSolution.txt";
    printSolution(sol, namesolution);

  };

///////////////////////////////////////////////////////////////////////////////////////////////////// Assemble Original

    void assemble_Original_Domain_and_Solution()
    {
        if (mpi_rank == 0 )
        {
            printDomain(dom_global, "GlobalDomain");
            for(int i = 0; i < DIM1*N/mpi_numproc; ++i)
                {
                    dom_global[i] = dom[i];
                    sol_global[i] = sol[i];
                }
            

            for(int i = 1; i < mpi_numproc; ++i)
            {
                // I send the complete vector and then pick what I need
                int additionalLayer = 2;
                if ( i == mpi_numproc-1)
                    additionalLayer = 1;
            
                int rec_domain[DIM1*(N+additionalLayer)];
                double rec_sol[DIM1*(N+additionalLayer)];
                MPI_Recv(rec_domain, DIM1*(N+additionalLayer), MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
                MPI_Recv(rec_sol, DIM1*(N+additionalLayer), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

                    for(int j = DIM1; j < DIM1*(N/mpi_numproc+1); ++j)
                    {
                        dom_global[DIM1*(N/mpi_numproc*i-1) + j] = rec_domain[j];
                        sol_global[DIM1*(N/mpi_numproc*i-1) + j] = rec_sol[j];
                    }

            }
        }
        if (mpi_rank != 0 )
        {
            string domainName = "GlobalPart" + std::to_string(mpi_rank) + ".txt";
            int send_domain[DIM1*DIM2];
            double send_sol[DIM1*DIM2];

            for(int i = 0; i < DIM1*DIM2; ++i)
            {
                send_domain[i] = dom[i];
                send_sol[i] = sol[i];
            }
            MPI_Send(send_domain, DIM1*DIM2, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(send_sol, DIM1*DIM2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
        

    };

/////////////////////////////////////////////////////////////////////////////////////////////////////////// print Domain

    void printDomain(std::vector<int> v, string name)
      {
        ofstream outfile;
        outfile.open(name, ios::out | ios::trunc);
        outfile << mpi_rank << endl << endl;
        std::vector<char> symbol_list { 'x', '#', 'o'};
        outfile << std::defaultfloat;
        for (int j = 0; j != v.size()/N; ++j)
        {
            outfile << j + mpi_rank*N/mpi_numproc - min(mpi_rank,1) << "\t";
          for (int i = 0; i != DIM1; ++i)
          {
            outfile << symbol_list[v[i + DIM1 * j]] << " ";
          }
          outfile << std::endl;
        }
        outfile << std::defaultfloat;
        outfile.close();
      };

////////////////////////////////////////////////////////////////////////////////////////////////////////// print Solution

void printSolution(std::vector<double> v, string name)
      {
        ofstream outfile;
        outfile.open(name, ios::out | ios::trunc);
        outfile << mpi_rank << endl << endl;
        outfile << std::defaultfloat;
        for (int j = 0; j != v.size()/N; ++j)
        {
            outfile << j + mpi_rank*N/mpi_numproc - min(mpi_rank,1) << "\t";
          for (int i = 0; i != N; ++i)
          {
            outfile << v[i + N * j] << " ";
          }
          outfile << std::endl;
        }
        outfile << std::defaultfloat;
        outfile.close();
      };
};

} // namespace nssc
