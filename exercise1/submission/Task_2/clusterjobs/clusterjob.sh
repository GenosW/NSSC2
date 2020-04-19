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
res = 250
int = 15

# the input-function does not work like this 
# to test the programm we wrote four different submitfiles

srun -n 1 ./jacobiMPI $res $int
srun -n 2 ./jacobiMPI $res $int
srun -n 4 ./jacobiMPI $res $int
srun -n 5 ./jacobiMPI $res $int
srun -n 8 ./jacobiMPI $res $int
srun -n 12 ./jacobiMPI $res $int
srun -n 16 ./jacobiMPI $res $int
srun -n 24 ./jacobiMPI $res $int
srun -n 32 ./jacobiMPI $res $int
srun -n 40 ./jacobiMPI $res $int
srun -n 50 ./jacobiMPI $res $int
srun -n 64 ./jacobiMPI $res $int
srun -n 80 ./jacobiMPI $res $int
