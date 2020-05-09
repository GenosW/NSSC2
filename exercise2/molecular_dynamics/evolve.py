'''evolve.py [path] [dt] [N]
takes an input initial situation and evolves from there using velocity Verlet algorithm

Takes 3 command line arguments (callable as above):
    path... path to input file
    dt... length of timestep
    N... number of time steps
'''
import jax.numpy as np
from jax import grad, jit
import numpy
from argparse import ArgumentParser
from simulation import Simulation_box, Epot_lj, grad_Epot, Verlet


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

    outfile = "snapshots/trajectory_dt0,0001.xyz"

    M, L, positions, velocities, description = Simulation_box.loadSnapshot(inputPath)
    Simulation_box.saveSnapshotStatic(positions, velocities, M, L, description, outfile, mode='w') # overwrite file
    for i in range(N-1):
        positions, velocities = Verlet(positions, velocities, dt, M, L)
        if i%10 == 0:
            print(f"Step {i} done")
            #positions = Simulation_box.enforceMI(positions, L)
        Simulation_box.saveSnapshotStatic(positions, velocities, M, L, description, outfile, mode='a')
