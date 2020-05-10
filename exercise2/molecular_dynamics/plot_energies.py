import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

energies_file = "snapshots/energies_dt0,03_p.txt"
snap_file = ("snapshots/trajectory_"+energies_file.split("_")[1]+"_"+energies_file.split("_")[2]).replace("txt","xyz") #"snapshots/trajectory_dt0,01_p.xyz"
dt_str = energies_file.split("_")[1].strip("dt")
analyzer = Simulation_Analyzer(snap_file)
Epots, Ekins, Etots = analyzer.loadEnergies(energies_file)

L = analyzer.sim.L
V = L*L*L
M = analyzer.sim.M
num_snaps = 1000
num_snaps_analyzed = int(0.75*1000)
norm_factor = V/(4*numpy.pi*M*M)
print((L, M, V, num_snaps_analyzed))

# Normalize
#print(y)
print(Epots.shape)

# Plot the result
a = 0
b = -1
fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
plt.plot(range(Epots[a:b].size), Epots[a:b], 'r-x', label="Epot")
plt.plot(range(Ekins[a:b].size), Ekins[a:b], 'b-x', label="Ekin")
plt.plot(range(Etots[a:b].size), Etots[a:b], 'g-x', label="Etot")
plt.title("Energies dt = "+dt_str)
plt.xlabel("iterations")
plt.ylabel("Energy")
# plt.xlim((0,30))
# plt.ylim
plt.legend()
pic_file = energies_file.replace("txt","png")
plt.savefig(pic_file)
