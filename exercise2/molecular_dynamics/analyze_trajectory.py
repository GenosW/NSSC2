'''analyze_trajectory.py [trajectory.xyz]

analyzes a trajectory file in .xyz-format. The path of the file is given as command line argument.
'''
import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

# Command-line argument parser
parser = ArgumentParser(
    description=
    'eanalyzes a trajectory file in .xyz-format. The path of the file is given as command line argument.'
)
parser.add_argument('path',
                    metavar='trajectory.xyz',
                    type=str,
                    help='a path to a trajectory file (.xyz);')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    inputPath = args.path

    analysis = Simulation_Analyzer(inputPath)
    x, y, rest = analysis.analyze(num_samples=100, start=0.25, stop=1) # start, stop of density calculation (0=0%, 0.25=25%, 1=100%, etc)
    print("M=", analysis.sim.M)
    print("L=", analysis.sim.L)
    print("L/2=", analysis.sim.L / 2)
    print("L/2*sqrt(3)=", analysis.sim.L / 2 * np.sqrt(3))
    print("Rest:", rest)

    dt = "0,03"
    analysis.savePCF(x, y, path="snapshots/pcf_dt"+dt+"_p.txt")
    analysis.saveEnergies(path="snapshots/energies_dt"+dt+"_p.txt")

