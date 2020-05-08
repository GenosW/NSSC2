'''analyze_trajectory.py [trajectory.xyz]

analyzes a trajectory file in .xyz-format. The path of the file is given as command line argument.
'''
import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Trajectory, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet


# Command-line argument parser
parser = ArgumentParser(description='eanalyzes a trajectory file in .xyz-format. The path of the file is given as command line argument.')
parser.add_argument('path', metavar='trajectory.xyz',type=str, help='a path to a trajectory file (.xyz);')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    inputPath = args.path

    analysis = Simulation_Analyzer(inputPath)
    x, y = analysis.analyze(start=0, stop=0.25)

    # Plot the result
    fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
    plt.plot(x, y, 'r-x', label="density")
    plt.title("Volumetric density")
    plt.legend()
    plt.show()
