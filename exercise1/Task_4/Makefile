# requirements on ubuntu
# sudo apt-get build-essentials
# sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# required modules on cluster
# module load mpi/openmpi-x86_64
# module load pmi/pmix-x86_64

CXX=g++
MPICXX?=mpic++
CXXFLAGS := $(CXXFLAGS) -std=c++14 -O3 -Wall -pedantic -march=native -ffast-math

.DEFAULT_GOAL := all
 
jacobiSERIAL: Makefile main.cpp solver.hpp arguments.hpp 
	$(CXX) main.cpp -o build/jacobiSERIAL $(CXXFLAGS)

jacobiMPI: Makefile main.cpp solver.hpp arguments.hpp
	$(MPICXX) main.cpp -o build/jacobiMPI2D -lpthread -DUSEMPI $(CXXFLAGS)

.PHONY: clean

clean:
	rm jacobiSERIAL jacobiMPI