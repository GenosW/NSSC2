#!/bin/bash
#SBATCH -J jacobiMPI
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --exclusive 
#SBATCH --time=00:40:00
#SBATCH --exclude=tcad[37]

if command -v sinfo  2>/dev/null # if on cluster
then
    module load mpi/openmpi-x86_64
    module load pmi/pmix-x86_64
    mpiproc=40
else  # if on local machine
    shopt -s expand_aliases
    alias srun='mpirun' 
    mpiproc=4
fi

srun -n 1 ./jacobiMPI 250 10
srun -n 2 ./jacobiMPI 250 10
srun -n 3 ./jacobiMPI 250 10
srun -n 4 ./jacobiMPI 250 10
srun -n 5 ./jacobiMPI 250 10
srun -n 6 ./jacobiMPI 250 10
srun -n 7 ./jacobiMPI 250 10
srun -n 8 ./jacobiMPI 250 10
srun -n 9 ./jacobiMPI 250 10
srun -n 10 ./jacobiMPI 250 10
srun -n 11 ./jacobiMPI 250 10
srun -n 12 ./jacobiMPI 250 10
srun -n 13 ./jacobiMPI 250 10
srun -n 14 ./jacobiMPI 250 10
srun -n 15 ./jacobiMPI 250 10
srun -n 16 ./jacobiMPI 250 10
srun -n 17 ./jacobiMPI 250 10
srun -n 18 ./jacobiMPI 250 10
srun -n 19 ./jacobiMPI 250 10
srun -n 20 ./jacobiMPI 250 10
srun -n 21 ./jacobiMPI 250 10
srun -n 22 ./jacobiMPI 250 10
srun -n 23 ./jacobiMPI 250 10
srun -n 24 ./jacobiMPI 250 10
srun -n 26 ./jacobiMPI 250 10
srun -n 27 ./jacobiMPI 250 10
srun -n 28 ./jacobiMPI 250 10
srun -n 29 ./jacobiMPI 250 10
srun -n 30 ./jacobiMPI 250 10
srun -n 31 ./jacobiMPI 250 10
srun -n 32 ./jacobiMPI 250 10
srun -n 33 ./jacobiMPI 250 10
srun -n 34 ./jacobiMPI 250 10
srun -n 35 ./jacobiMPI 250 10
srun -n 36 ./jacobiMPI 250 10
srun -n 37 ./jacobiMPI 250 10
srun -n 38 ./jacobiMPI 250 10
srun -n 39 ./jacobiMPI 250 10
srun -n 40 ./jacobiMPI 250 10





