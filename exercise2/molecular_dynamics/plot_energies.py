import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

analyzer = Simulation_Analyzer("snapshots/trajectory_dt0,03.xyz")
Epots, Ekins, Etots = analyzer.loadEnergies("snapshots/energies0,03.txt")

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
fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
plt.plot(range(Epots.size), Epots, 'r-x', label="Epot")
plt.plot(range(Ekins.size), Ekins, 'b-x', label="Ekin")
plt.plot(range(Etots.size), Etots, 'g-x', label="Etot")
plt.title("Energies dt = 0.03")
plt.xlabel("iterations")
plt.ylabel("Energy")
plt.legend()
plt.savefig("snapshots/energies0,03.png")
