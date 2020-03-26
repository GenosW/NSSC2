#!/bin/bash
#SBATCH -J jacobiMPI
#SBATCH -N 2
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --exclusive 

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

srun -n $mpiproc ./jacobiMPI 16 10000

