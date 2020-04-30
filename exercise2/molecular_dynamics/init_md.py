'''init_md.py [M] [L] [Sigma]
generates a ”relaxed” (i.e., low-energy) starting confguration for a molecular dynamics simulation.

Takes 3 command line arguments (callable as above):
    M... a number of particles;
    L... a side length for the simulation box; and
    Sigma... a standard deviation for the velocity distribution
'''
from scipy.optimize import minimize
from numpy.random import multivariate_normal
from argparse import ArgumentParser
from lennardjones import Simulation_box, Trajectory

# Command-line argument parser
parser = ArgumentParser(description='generates a ”relaxed” (i.e., low-energy) starting confguration for a molecular dynamics simulation.')
parser.add_argument('M', metavar='M...',type=int, help='a number of particles;')
parser.add_argument('L', metavar='L...',type=float, help='a side length for the simulation box;')
parser.add_argument('Sigma', metavar='Sigma...',type=float, help='a standard deviation for the velocity distribution')

if __name__ == "__main__":
    args = parser.parse_args()
    M = args.M
    L = args.L
    Sigma = args.Sigma
    print("Initializing molecular dynamics simulation with:")
    print(f"M = {M}")
    print(f"L = {L}")
    print(f"Sigma = {Sigma}")
    sim = Simulation_box(M, L)
    print(sim.molecules)