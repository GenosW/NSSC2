import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

parser = ArgumentParser(description='plots a pcf.txt file')
parser.add_argument('path', metavar='energies.txt',type=str, help='a path to a pcf file;', default="snapshots/energies.txt")

# Parse arguments
args = parser.parse_args()
print(args)
path = args.path

energies_file = path
snap_file = path.replace("energies", "trajectory") #"snapshots/trajectory_dt0,01_p.xyz"
try:
    dt_str = energies_file.split("_")[1].strip("dt")
except:
    dt_str = "???"
analyzer = Simulation_Analyzer(snap_file)
Epots, Ekins, Etots = analyzer.loadEnergies(energies_file)

L = analyzer.sim.L
V = L*L*L
M = analyzer.sim.M
num_snaps = 1000
num_snaps_analyzed = int(0.75*1000)
norm_factor = V/(4*numpy.pi*M*M)
print((L, M, V, num_snaps_analyzed))


# Plot the result
fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
plt.plot(range(Epots.size), Epots, 'r-x', label="Epot")
plt.plot(range(Ekins.size), Ekins, 'b-x', label="Ekin")
plt.plot(range(Etots.size), Etots, 'g-x', label="Etot")
plt.title("Energies dt = "+dt_str)
plt.xlabel("iterations")
plt.ylabel("Energy")
plt.legend()
figpath = path.replace("xyz", "png") #"snapshots/vol_dens_dt0,03p.png"
plt.savefig(figpath)
print(f"Saved plot in <{figpath}")