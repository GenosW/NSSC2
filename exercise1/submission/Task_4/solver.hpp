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
#include <numeric>
//#ifdef USEMPI
#include <mpi.h>
//#include <mpi.h>
//#endif
namespace nssc
{

enum Cell
{
  UNKNOWN = 0,
  DIR = 1,
  GHOST = 2
};
template <typename P=double>
struct Discretization
{
  Discretization(size_t N)
      : resolution(N), h(1.0 / (resolution - 1)),
        C(4.0 / (h * h) + 4 * M_PI * M_PI), N(-1.0 / (h * h)),
        S(-1.0 / (h * h)), W(-1.0 / (h * h)), E(-1.0 / (h * h)) {}
  const int resolution;
  const P h, C, N, S, W, E;
};

template <typename P=double>
P Solution(P x, P y)
{
  return sin(2 * M_PI * x) * sinh(2 * M_PI * y);
}

// 2D field with domain info and ghost layer
template <typename P=double>
class Field
{
public:
  MPI_Status stat;
  const Discretization<P> disc;
  int resolution, DIM1, DIM2, M, N, m, n, decomp_mode;

  std::vector<P> sol;  // local solution
  std::vector<P> sol2; // local solution swap
  std::vector<P> rhs;  // local rhs
  std::vector<int> dom;     // local domaininfo
  // COM arrays/vectors
  // vertical communication
  std::vector<P> msg_upper;
  std::vector<P> rec_upper;
  std::vector<P> msg_lower;
  std::vector<P> rec_lower;
  // horizontal communication
  std::vector<P> msg_left;
  std::vector<P> rec_left;
  std::vector<P> msg_right;
  std::vector<P> rec_right;
  P norm2_residual;
  P normMax_residual;
  int numberiterations;
  bool debugmode = false;           // enable to get a bunch of extra information about local domains, local solutions etc. in text files
  bool comparemode = false;         // enable to generate compare file for Task 3
  string name;
  std::vector<P> results; // results array

  int mpi_rank;
  int mpi_numproc;
  int rank_upperNeighbor, rank_lowerNeighbor, rank_leftNeighbor, rank_rightNeighbor;
  MPI_Datatype PREC_MPI;

  ~Field(){ }
  Field(int resolution ,int rank, int numproc, int mode) : disc(resolution), resolution(resolution), decomp_mode(mode), mpi_rank(rank), mpi_numproc(numproc)
  {
    if (std::is_same<P, float>::value)
    {
      PREC_MPI = MPI_FLOAT;
    }
    else
    {
      PREC_MPI = MPI_DOUBLE;
    }

    if(debugmode)
    {
        std::cout << "Rank " << mpi_rank << std::endl;
        name = "local_Domain_rank" + std::to_string(mpi_rank) + ".txt";
    }
    assert(mode==1 or mode==2);
    // allocate arrays
    M = 1; // num of proc in x
    N = 1; // num of proc in y
    if (decomp_mode == 1)
    {
      M = 1; // num of proc in x
      N = mpi_numproc; // num of proc in y
    }
    else if (decomp_mode == 2)
    {
      factorization(mpi_numproc);
    }
    else
    {
      cout << "Unknown factorization mode! --> Terminating...";
      if (mpi_numproc > 1){
        MPI_Finalize();
      }
      exit(1);
    }
    
    
    if(mpi_rank == 0)
    {
      if (decomp_mode == 1)
      {
        std::cout << endl << "Calculation of " << resolution << "x" << resolution << " Grid with " << mpi_numproc << " process(es) using 1D decomposition" << endl; 
      }
      else if (decomp_mode == 2)
      {
        std::cout << endl << "Calculation of " << resolution << "x" << resolution << " Grid with " << mpi_numproc << " process(es) using 2D decomposition" << endl 
        << "M: " << M << ", N: " << N << endl;
      }
    }
    m = mpi_rank%M;
    n = (mpi_rank-m)/M;

    // Check how many additional (ghost) layers are needed per axis
    int additionalLayer_X = 2;
    int additionalLayer_Y = 2;
    if ( m == 0 || m == M-1 ){
      additionalLayer_X = 1;
    }
    if ( M == 1 ){
      additionalLayer_X = 0;
    }
    if ( n == 0 || n == N-1 ){
      additionalLayer_Y = 1;
    }
    if (mpi_numproc == 1){
      additionalLayer_Y = 0;
    }
    rank_upperNeighbor = (n-1)*M+m;
    rank_lowerNeighbor = (n+1)*M+m;
    rank_leftNeighbor = n*M+m-1;
    rank_rightNeighbor = n*M+m+1;

    if ( m <= M-2 ){
        DIM1 = (int)std::floor((double)resolution/M) + additionalLayer_X;
    }
    else{
        DIM1 = resolution - (int)std::floor((double)resolution/M)*(M-1) + additionalLayer_X;
    }
    if ( n <= N-2 ){
        DIM2 = (int)std::floor((double)resolution/N) + additionalLayer_Y;
    }
    else{
        DIM2 = resolution - (int)std::floor((double)resolution/N)*(N-1) + additionalLayer_Y;
    }

    sol = std::vector<P>(DIM1 * DIM2, 0);
    sol2 = std::vector<P>(DIM1 * DIM2, 0);
    rhs = std::vector<P>(DIM1 * DIM2, 0);
    dom = std::vector<int>(DIM1 * DIM2, Cell::UNKNOWN);
	  results = std::vector<P>(3, 0);
    // vertical communication
    msg_upper = std::vector<P>(DIM1, 0);
    rec_upper= std::vector<P>(DIM1, 0);
    msg_lower = std::vector<P>(DIM1, 0);
    rec_lower = std::vector<P>(DIM1, 0);
    // horizontal communication
    msg_left = std::vector<P>(DIM2, 0);
    rec_left = std::vector<P>(DIM2, 0);
    msg_right = std::vector<P>(DIM2, 0);
    rec_right = std::vector<P>(DIM2, 0);

///////////////////////////////////////////////////////////////////////////////////////////// setup local domain

    for (int j = 0; j != DIM2; ++j)
    {
      for (int i = 0; i != DIM1; ++i)
      {
        if ( i==0 || i==DIM1-1 || j==0 || j==DIM2-1 ) // ghost layer
          dom[i + DIM1 * j] = Cell::GHOST;

        if ( (m==0 && i==0 && n==0 && j!=DIM2-1) || (m==0 && i==0 && n!=0 && n!=N-1 && j!=0 && j!=DIM2-1) || (m==0 && i==0 && n==N-1 && j!=0) || 
             (m==M-1 && i==DIM1-1 && n==0 && j!=DIM2-1) || (m==M-1 && i==DIM1-1 && n!=0 && n!=N-1 && j!=0 && j!=DIM2-1) || (m==M-1 && i==DIM1-1 && n==N-1 && j!=0) ||
             (n==0 && j==0 && m==0 && i!=DIM1-1) || (n==0 && j==0 && m!=0 && m!=M-1 && i!=DIM1-1 && i!=0) || (n==0 && j==0 && m==M-1 && i!=0) ||
             (n==N-1 && j==DIM2-1 && m==0 && i!=DIM1-1) || (n==N-1 && j==DIM2-1 && m!=0 && m!=M-1 && i!=DIM1-1 && i!=0) || (n==N-1 && j==DIM2-1 && m==M-1 && i!=0) )// global domain boundary
          dom[i + DIM1 * j] = Cell::DIR;
      }
    }
    if(debugmode)
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
    if(debugmode)
    {
     name = "initialRhs_rank_" + std::to_string(mpi_rank) + ".txt";
     printLocalRhs(name);
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
    if(debugmode)
    {
      name = "initialSolution_rank_" + std::to_string(mpi_rank) + ".txt";
      printLocalSolution(name);
    }
  }
/////////////////////////////////////////////////////////////////////////////////////////////// perform Jacobi Iteration, with optional skip range

  void solve(int iterations)
  {

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    P runtime, totalComTime=0;

    if (mpi_rank == 0)
        start = std::chrono::high_resolution_clock::now();

    int iter;
    for (iter = 1; iter <= iterations; ++iter)
    {
      totalComTime += update();
    }
    numberiterations = iter-1;

    if (mpi_rank == 0)
    {
      end = std::chrono::high_resolution_clock::now();
      runtime =
          std::chrono::duration_cast<std::chrono::duration<P>>(end - start)
              .count();

      std::cout << endl << std::scientific << "runtime " << runtime << std::endl;
      std::cout << std::scientific << "runtime/iter " << runtime / iter << std::endl;
      std::cout << endl << std::scientific << "com time " << totalComTime << std::endl;
      results[0] = runtime;
      results[1] = runtime / iter;
      results[2] = totalComTime;
    }
    //assemble_Original_Domain_and_Solution();
  }

//////////////////////////////////////////////////////////////////////////////////////////////// update local solution

  P update()
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

    P comTime = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> startCom;
    if (mpi_rank == 0) {startCom = std::chrono::high_resolution_clock::now();};
    ///////////////////////// vertical communication /////////////////////////////////
    MPI_Request req_upper, req_lower, req_left, req_right;
    if (n > 0) // n...position of process/decomposition in vertical dimension
    {
      for (int i = 0; i < DIM1; ++i)
          msg_upper[i] = sol[DIM1+i];
      
      MPI_Isend(msg_upper.data(), DIM1, PREC_MPI, // triplet of buffer, size, data type
                rank_upperNeighbor, // target: (n-1)*M+m,upper neighbor
                1,
                MPI_COMM_WORLD,
                &req_upper);
    }
    if (n < N-1)
    {
      MPI_Recv(rec_upper.data(), DIM1, PREC_MPI,
              rank_lowerNeighbor, // target: (n+1)*M+m --> lower neighbor
              1, //MPI_ANY_TAG,
              MPI_COMM_WORLD, 
              &stat);
      for (int i = 0; i < DIM1; ++i)
          sol[(DIM2-1)*DIM1+i] = rec_upper[i];
    }
    if (n > 0 && rank_upperNeighbor >= 0) // n...position of process/decomposition in vertical dimension
    {
        MPI_Wait(&req_upper, &stat); // Is upper vert comm done?
    }
    // If upper vert comm done --> do lower vert communication
    if (n < N-1)
    {
      for (int i = 0; i < DIM1; ++i)
          msg_lower[i] = sol[DIM1*(DIM2-2)+i];
      
      MPI_Isend(msg_lower.data(), DIM1, PREC_MPI, 
              rank_lowerNeighbor, // (n+1)*M+m,
              2,
              MPI_COMM_WORLD,
              &req_lower);
    }
    if (n > 0)
    {
      MPI_Recv(rec_lower.data(), DIM1, PREC_MPI,
                rank_upperNeighbor,
                2, //MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &stat);
      for (int i = 0; i < DIM1; ++i)
          sol[i] = rec_lower[i];
    }
    if (n < N-1)
    {
        MPI_Wait(&req_lower, &stat); // Is lower vert comm done?
    }
    if (decomp_mode > 1){
      // If lower + upper vert comm done --> do horizontal communication
      ///////////////////////// horizontal communication /////////////////////////////////
      // left horizontal comm
      if (m > 0)
      {
        for (int j = 0; j < DIM2; ++j)
            msg_left[j] = sol[DIM1*j+1];
        
        MPI_Isend(msg_left.data(), DIM2, PREC_MPI,
                  rank_leftNeighbor, // n*M+m-1,
                  3,
                  MPI_COMM_WORLD,
                  &req_left);//, req_left);
      }
      if (m < M-1)
      {
        MPI_Recv(rec_left.data(), DIM2, PREC_MPI, 
                  rank_rightNeighbor, // n*M+m+1,
                  3,//MPI_ANY_TAG,
                  MPI_COMM_WORLD,
                  &stat);
        for (int j = 0; j < DIM2; ++j)
            sol[(DIM1-1)+j*DIM1] = rec_left[j];
      }
      if (m > 0){
        MPI_Wait(&req_left, &stat); // Is left hori comm done?
      }
      //If left hori comm done --> do right horizontal communication
      if (m < M-1)
      {
        for (int j = 0; j < DIM2; ++j)
            msg_right[j] = sol[(DIM1-2)+j*DIM1];
        
        MPI_Isend(msg_right.data(), DIM2, PREC_MPI, 
                  rank_rightNeighbor,
                  4,
                  MPI_COMM_WORLD,
                  &req_right);
      }
      if (m > 0)
      {
        MPI_Recv(rec_right.data(), DIM2, PREC_MPI,
                  rank_leftNeighbor,
                  4,//MPI_ANY_TAG,
                  MPI_COMM_WORLD,
                  &stat);
        for (int j = 0; j < DIM2; ++j)
            sol[j*DIM1] = rec_right[j];
      }
      if (m < M-1){
        MPI_Wait(&req_right, &stat); // Is right hori comm done? --> Is all comm done?
      }
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> endCom;
    if (mpi_rank == 0) {
      endCom = std::chrono::high_resolution_clock::now();
      comTime = std::chrono::duration_cast<std::chrono::duration<P>>(endCom - startCom).count();
    }
    if(debugmode)
    {
      string name = "localSolution" + std::to_string(mpi_rank) + "__" + std::to_string(mpi_numproc) + ".txt";
      printLocalSolution(name);
    }
    return comTime;
  }

/////////////////////////////////////////////////////////////////////////////////////// calculate residual locally 

  void residual_local()
  {
    if ( mpi_rank == 0)
    {
      P max = 0;
      P sum = 0;
      int count = 0;
      for (int j = 0; j != DIM2; ++j)
      {
        for (int i = 0; i != DIM1; ++i)
        {
          if (dom[i + DIM1 * j] == Cell::UNKNOWN)
          {
            P tmp = Solution(real_x(i) * disc.h, real_y(j) * disc.h) * 4 * M_PI * M_PI -
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
      for (int k = 1; k < mpi_numproc; k++)
      {
          P max_recieved;
          P sum_recieved;
          MPI_Recv(&max_recieved, 1, PREC_MPI, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
          MPI_Recv(&sum_recieved, 1, PREC_MPI, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
          max = fabs(max_recieved) > max ? fabs(max_recieved) : max;
          sum += sum_recieved;
      }
      norm2_residual = sqrt(sum);
      normMax_residual = max;

      std::cout << endl;
      std::cout << std::scientific << "norm2res_gloabl: " << norm2_residual << std::endl;
      std::cout << std::scientific << "normMres_global: " << normMax_residual << std::endl;
      // results[4] = norm2_residual;
      // results[5] = normMax_residual;
    }
    else
    {
      P max = 0;
      P sum = 0;
      int count = 0;
      for (int j = 0; j != DIM2; ++j)
      {
        for (int i = 0; i != DIM1; ++i)
        {
          if (dom[i + DIM1 * j] == Cell::UNKNOWN)
          {
            P tmp = Solution(real_x(i) * disc.h, real_y(j) * disc.h) * 4 * M_PI * M_PI -
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
      MPI_Send(&max, 1, PREC_MPI, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&sum, 1, PREC_MPI, 0, 1, MPI_COMM_WORLD);
    }

  }




/////////////////////////////////////////////////////////////////////////////////////// calculate error locally 

  void error_local()
  {
    if ( mpi_rank == 0)
    {
      P max = 0;
      P sum = 0;
      for (int j = 0; j != DIM2; ++j)
      {
        for (int i = 0; i != DIM1; ++i)
        {
          if (dom[i + DIM1 * j] == Cell::UNKNOWN)
          {
            P tmp = sol[i +DIM1 * j] -
                          Solution(real_x(i) * disc.h, real_y(j) * disc.h);

            max = fabs(tmp) > max ? fabs(tmp) : max;
            sum += tmp * tmp;
          }
        }
      }
      for (int k = 1; k < mpi_numproc; k++)
      {
          P max_recieved;
          P sum_recieved;
          MPI_Recv(&max_recieved, 1, PREC_MPI, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
          MPI_Recv(&sum_recieved, 1, PREC_MPI, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
          max = fabs(max_recieved) > max ? fabs(max_recieved) : max;
          sum += sum_recieved;
      }
      P norm2 = sqrt(sum);
      P normMax = max;

      std::cout << endl;
      std::cout << std::scientific << "norm2err_global: " << norm2 << std::endl;
      std::cout << std::scientific << "normMerr_global: " << normMax << std::endl;

      if(comparemode)
      {
          ofstream outfile;
          outfile.open("Compare_double.txt", fstream::app);
          outfile << std::defaultfloat;
          outfile << resolution << "   " << numberiterations << "   " << norm2_residual << "   " << normMax_residual << "   " << norm2 << "   " << normMax << endl;
          outfile.close(); 
      }
    }
    else
    {
      P max = 0;
      P sum = 0;
      for (int j = 0; j != DIM2; ++j)
      {
        for (int i = 0; i != DIM1; ++i)
        {
          if (dom[i + DIM1 * j] == Cell::UNKNOWN)
          {
            P tmp = sol[i +DIM1 * j] -
                          Solution(real_x(i) * disc.h, real_y(j) * disc.h);

            max = fabs(tmp) > max ? fabs(tmp) : max;
            sum += tmp * tmp;
          }
        }
      }
      MPI_Send(&max, 1, PREC_MPI, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&sum, 1, PREC_MPI, 0, 1, MPI_COMM_WORLD);
    }

  }
////////////////////////////////////////////////////////////////////////////////////////////// write resultvector
  void printResults( string name){
		if (mpi_rank == 0)
    	{
			string file_name = "data/"+name+".csv";
		    ofstream outfile;
			outfile.open (file_name);
			outfile << "runtime,runtime/iter,totalComTime\n";
			outfile << results[0] << "," << results[1] << "," <<  results[2] << std::endl;
			outfile << std::defaultfloat;
		    outfile.close();
		}
 	  };
////////////////////////////////////////////////////////////////////////////////////////////// print local Domain (debugging)

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



////////////////////////////////////////////////////////////////////////////////////////////// print local Solution (debugging)

  void printLocalSolution( string name)
    {
      ofstream outfile;
      outfile.open(name, ios::out | ios::trunc);
      outfile << "Rank " << mpi_rank << ", m: " << m << ", n: " << n << endl << endl;
      outfile << "\t";
      outfile << std::defaultfloat;
      for (int i = 0; i < DIM1; i++)
          outfile << real_x(i) << "\t";
      outfile << endl;
      for (int j = 0; j < DIM2; ++j)
      {
          outfile << real_y(j) << "\t";
        for (int i = 0; i < DIM1; ++i)
        {
          outfile << sol[i + DIM1 * j] << "\t";
        }
        outfile << std::endl;
      }
      outfile << std::defaultfloat;
      outfile.close();   
    };

////////////////////////////////////////////////////////////////////////////////////////////// print local Rhs (debugging)

  void printLocalRhs( string name)
    {
      ofstream outfile;
      outfile.open(name, ios::out | ios::trunc);
      outfile << "Rank " << mpi_rank << ", m: " << m << ", n: " << n << endl << endl;
      outfile << "\t";
      outfile << std::defaultfloat;
      for (int i = 0; i < DIM1; i++)
          outfile << real_x(i) << "\t";
      outfile << endl;
      for (int j = 0; j < DIM2; ++j)
      {
          outfile << real_y(j) << "\t";
        for (int i = 0; i < DIM1; ++i)
        {
          outfile << rhs[i + DIM1 * j] << "\t";
        }
        outfile << std::endl;
      }
      outfile << std::defaultfloat;
      outfile.close();   
    };

////////////////////////////////////////////////////////////////////////////////////////////////// real X and Y

int real_x(int i)
  {
    int x = i + m*(int)std::floor((P)resolution/(P)M) - min(m,1);
    return x;
  };

int real_y(int j)
  {
    int y = j + n*(int)std::floor((P)resolution/(P)N) - min(n,1);
    return y;
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
      
      //int factors[int(c)];
      std::vector<int> factors = std::vector<int>(int(c));
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

//////////////////////////////////////////////////////////////////////////////////////////////////////// printArray (not used)

  void printArray(std::vector<P> &v){
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
/*
/////////////////////////////////////////////////////////////////////////////////////// calculate residual global (not used)

  void residual_global()
  {
    if ( mpi_rank == 0)
    {
    P max = 0;
    P sum = 0;
    int count = 0;
    for (int j = 0; j != resolution; ++j)
    {
      for (int i = 0; i != resolution; ++i)
      {
        if (dom_global[i + resolution * j] == Cell::UNKNOWN)
        {
          P tmp = Solution(i * disc.h, j * disc.h) * 4 * M_PI * M_PI -
                       (sol_global[(i + 0) + resolution * (j - 0)] * disc.C +
                        sol_global[(i + 1) + resolution * (j - 0)] * disc.E +
                        sol_global[(i - 1) + resolution * (j - 0)] * disc.W +
                        sol_global[(i + 0) + resolution * (j - 1)] * disc.S +
                        sol_global[(i + 0) + resolution * (j + 1)] * disc.N);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
          ++count;
        }
      }
    }
	end_jac = std::chrono::high_resolution_clock::now();
	runtime += std::chrono::duration_cast<std::chrono::duration<P>>(end_jac - start_jac).count();
	return runtime;

    P norm2 = sqrt(sum);
    P normMax = max;

    std::cout << endl;
    std::cout << std::scientific << "norm2res_gloabl: " << norm2 << std::endl;
    std::cout << std::scientific << "normMres_global: " << normMax << std::endl;
    }
  };

///////////////////////////////////////////////////////////////////////////////////////////////////// Assemble Original (not used)

  void assemble_Original_Domain_and_Solution()
    {
        if (mpi_rank == 0 )
        {

            for (int j = 0; j < DIM2; j++)
            {
                for (int i = 0; i < DIM1; i++)
                {
                    if (dom[j*DIM1 + i] != Cell::GHOST)
                    {
                        dom_global[j*resolution+i] = dom[j*DIM1 + i];
                        sol_global[j*resolution+i] = sol[j*DIM1 + i];
                    }
                }
            }
            for(int k = 1; k < mpi_numproc; k++)
            {
                // I send the complete vector and then pick what I need
                int global_x_coord;
                int global_y_coord;
                int DIM1p, DIM2p;
                int mp = k%M;
                int np = (k-mp)/M;
                int additionalLayer_Xp = 2;
                int additionalLayer_Yp = 2;
                if ( mp == 0 || mp == M-1 )
                    additionalLayer_Xp = 1;
                if ( M == 1 )
                    additionalLayer_Xp = 0;
                if ( np == 0 || np == N-1 )
                    additionalLayer_Yp = 1;
                if (mpi_numproc == 1)
                    additionalLayer_Yp = 0;
                
                if ( mp <= M-2 )
                    DIM1p = (int)std::floor((P)resolution/M) + additionalLayer_Xp;
                else
                    DIM1p = resolution - (int)std::floor((P)resolution/M)*(M-1) + additionalLayer_Xp;
                if ( np <= N-2 )
                    DIM2p = (int)std::floor((P)resolution/N) + additionalLayer_Yp;
                else
                    DIM2p = resolution - (int)std::floor((P)resolution/N)*(N-1) + additionalLayer_Yp;
                if(debugmode)
                    std::cout << "Rank " << k << " mp: " << mp << " np: " << np << " DIM1p: " << DIM1p << " DIM2p: " << DIM2p << endl;
                
                int rec_domain[DIM1p*DIM2p];
                P rec_sol[DIM1p*DIM2p];
                MPI_Recv(rec_domain, DIM1p*DIM2p, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
                MPI_Recv(rec_sol, DIM1p*DIM2p, PREC_MPI, k, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

                for ( int j = 0; j < DIM2p; j++)
                {
                    for ( int i = 0; i < DIM1p; i++)
                    {
                        if (rec_domain[j*DIM1p + i] != Cell::GHOST)
                        {
                            global_x_coord = i + mp*(int)std::floor((P)resolution/(P)M) - min(mp,1);
                            global_y_coord = j + np*(int)std::floor((P)resolution/(P)N) - min(np,1);
                            dom_global[ global_y_coord*resolution + global_x_coord ] = rec_domain[j*DIM1p + i];
                            sol_global[ global_y_coord*resolution + global_x_coord ] = rec_sol[j*DIM1p + i];
                        }
                    }
                }
            }
            if(debugmode)
            {
                string gd = "GlobalDomain_" + std::to_string(mpi_numproc) + ".txt";
                string gs = "GlobalSolution_" + std::to_string(mpi_numproc) + ".txt";
                printGlobalDomain(gd);
                printGlobalSolution(gs);
            }
                
        }

        if (mpi_rank != 0 )
        {
            int send_domain[DIM1*DIM2];
            P send_sol[DIM1*DIM2];

            for ( int j = 0; j < DIM2; j++)
            {
                for ( int i = 0; i < DIM1; i++)
                {
                    send_domain[j*DIM1+i] = dom[j*DIM1+i];
                    send_sol[j*DIM1+i] = sol[j*DIM1+i];
                }
            }

            MPI_Send(send_domain, DIM1*DIM2, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(send_sol, DIM1*DIM2, PREC_MPI, 0, 1, MPI_COMM_WORLD);
        }
        

    };

//////////////////////////////////////////////////////////////////////////////////////////////////////// calculate error global (not used)

  void error_global()
  {
    if (mpi_rank == 0)
    {
    P max = 0;
    P sum = 0;
    for (int j = 0; j != resolution; ++j)
    {
      for (int i = 0; i != resolution; ++i)
      {
        if (dom_global[i + resolution * j] == Cell::UNKNOWN)
        {
          P tmp = sol_global[i +resolution * j] -
                       Solution(i * disc.h, j * disc.h);

          max = fabs(tmp) > max ? fabs(tmp) : max;
          sum += tmp * tmp;
        }
      }
    }

    P norm2 = sqrt(sum);
    P normMax = max;

    std::cout << endl;
    std::cout << std::scientific << "norm2err_global: " << norm2 << std::endl;
    std::cout << std::scientific << "normMerr_global: " << normMax << std::endl;

    }
  };
////////////////////////////////////////////////////////////////////////////////////////////// print global Domain (debugging)

  void printGlobalDomain( string name)
      {
        ofstream outfile;
        outfile.open(name, ios::out | ios::trunc);
        outfile << "Rank " << mpi_rank << ", m: " << m << ", n: " << n << endl << endl;
        outfile << "\t";
        std::vector<char> symbol_list { 'x', '#', 'o'};
        outfile << std::defaultfloat;
        for (int i = 0; i < resolution; i++)
            outfile << i << "\t";
        outfile << endl;
        for (int j = 0; j < resolution; ++j)
        {
            outfile << j << "\t";
          for (int i = 0; i < resolution; ++i)
          {
            outfile << symbol_list[dom_global[i + resolution * j]] << "\t";
          }
          outfile << std::endl;
        }
        outfile << std::defaultfloat;
        outfile.close();   
      };

////////////////////////////////////////////////////////////////////////////////////////////// print global Solution (debugging)

  void printGlobalSolution( string name)
      {
        ofstream outfile;
        outfile.open(name, ios::out | ios::trunc);
        outfile << "Rank " << mpi_rank << ", m: " << m << ", n: " << n << endl << endl;
        outfile << "\t";
        outfile << std::defaultfloat;
        for (int i = 0; i < resolution; i++)
            outfile << real_x(i) << "\t";
        outfile << endl;
        for (int j = 0; j < resolution; ++j)
        {
            outfile << real_y(j) << "\t";
          for (int i = 0; i < resolution; ++i)
          {
            outfile << sol_global[i + resolution * j] << "\t";
          }
          outfile << std::endl;
        }
        outfile << std::defaultfloat;
        outfile.close();   
      };
*/
};


} // namespace nssc
