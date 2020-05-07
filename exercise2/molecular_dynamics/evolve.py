'''init_md.py [M] [L] [Sigma]
generates a ”relaxed” (i.e., low-energy) starting confguration for a molecular dynamics simulation.

Takes 3 command line arguments (callable as above):
    M... a number of particles;
    L... a side length for the simulation box; and
    Sigma... a standard deviation for the velocity distribution
'''
import jax.numpy as np
from jax import grad, jit
import numpy
from argparse import ArgumentParser
from simulation import Simulation_box, Trajectory, Epot_lj, grad_Epot, Verlet


# Command-line argument parser
parser = ArgumentParser(description='evolves an initial configuration of a molecular dynamics system, given in the input file <input.xyz>, and evolves it over time via the Verlet algorithm.')
parser.add_argument('path', metavar='input.xyz',type=str, help='a path to a snapshot file (.xyz);')
parser.add_argument('dt', metavar='dt',type=float, help='time step length;')
parser.add_argument('N', metavar='N',type=int, help='number of steps to perform.')
parser.add_argument('-name', metavar='name', type=str, help='Name/ID of simulation that is, if given, prepended to the description.', default="")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    inputPath = args.path
    dt = args.dt
    N = args.N
    name = args.name

    M, L, positions, velocities, description = Simulation_box.loadSnapshot(inputPath)
    Simulation_box.saveSnapshotStatic(positions, velocities, M, L, description, 'trajectory.txt', mode='a')
    for i in range(N-1):
        positions, velocities = Verlet(positions, velocities, dt, M, L)
        Simulation_box.saveSnapshotStatic(positions, velocities, M, L, description, 'trajectory.txt', mode='a')
