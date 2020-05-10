#!/bin/bash
#SBATCH -J jacobiMPI
#SBATCH -N 2
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --exclude=tcad[37]

if command -v sinfo  2>/dev/null # if on cluster
then
    module load mpi/openmpi-x86_64
    module load pmi/pmix-x86_64
    mpiproc=80
else  # if on local machine
    shopt -s expand_aliases
    alias srun='mpirun' 
    mpiproc=4
fi
res = 8000
int = 15

srun -n 1 ./jacobiMPI 8000 15
srun -n 2 ./jacobiMPI 8000 15
srun -n 4 ./jacobiMPI 8000 15
srun -n 5 ./jacobiMPI 8000 15
srun -n 8 ./jacobiMPI 8000 15
srun -n 12 ./jacobiMPI 8000 15
srun -n 16 ./jacobiMPI 8000 15
srun -n 24 ./jacobiMPI 8000 15
srun -n 32 ./jacobiMPI 8000 15
srun -n 40 ./jacobiMPI 8000 15
srun -n 50 ./jacobiMPI 8000 15
srun -n 64 ./jacobiMPI 8000 15
srun -n 80 ./jacobiMPI 8000 15