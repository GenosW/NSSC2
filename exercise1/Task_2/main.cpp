#include <assert.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string.h>
#ifdef USEMPI
#include <mpi.h>
#endif
#include "arguments.hpp"
#include "solver.hpp"

int main(int argc, char *argv[])
{
  int rank=0;
  int numproc=1;
  string name;
  bool comparemode = false; // for generating compare files for Task 3
#ifdef USEMPI
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  using namespace nssc;

  auto resolution = convertTo<int>(1, 1000, argc, argv);
  auto iterations = convertTo<int>(2, 30, argc, argv);

  assert(iterations > 0);
  assert(resolution > 3);
  if(!comparemode)
  {
      auto field = Field(resolution,rank,numproc);

      name = to_string(resolution)+","+"Num_of_proc"+","+to_string(numproc);

      field.solve(iterations);
      field.residual_local();
      field.error_local();
      field.printresults(name);
  }
  if(comparemode)
  {
    for (int i = 6; i <= 9; i++)
    {
        for (int j = 0; j <= 7; j++)
        {
            auto field = Field(std::pow(2,i),rank,numproc);
            field.solve(std::pow(10,j));
            field.residual_local();
            field.error_local();
        }
    }
  }

#ifdef USEMPI
  MPI_Finalize();
#endif

  return 0;

}
