import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

analyzer = Simulation_Analyzer("snapshots/trajectory.xyz")
r,y = analyzer.loadPCF("snapshots/pcf.txt")

L = analyzer.sim.L
V = L*L*L
M = analyzer.sim.M
num_snaps = 1000
num_snaps_analyzed = int(0.75*1000)
norm_factor = V/(4*numpy.pi*M*M)
print((M, L, V, num_snaps_analyzed))

# Normalize
#print(y)
y *= norm_factor /num_snaps_analyzed
y[1:] /= (r[1:]*r[1:])
print(y.shape)

# Plot the result
fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
plt.plot(r, y, 'r-x', label="density")
plt.title("Volumetric density")
plt.xlabel("r... distance to particle in origin")
plt.ylabel("Volumetric density")
plt.legend()
plt.savefig("snapshots/vol_dens.png")