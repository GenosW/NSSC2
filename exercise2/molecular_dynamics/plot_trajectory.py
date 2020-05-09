import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet

parser = ArgumentParser(description='plots a pcf.txt file')
parser.add_argument('path', metavar='trajectory.xyz',type=str, help='a path to a trajectory file (.xyz);')
parser.add_argument('num', metavar='num',type=str, help='number of snaps analyzed --> take average')

# Parse arguments
args = parser.parse_args()
print(args)
path = args.path
num_snaps_analyzed = int(args.num)

analyzer = Simulation_Analyzer("snapshots/trajectory_laura.xyz")
r,y = analyzer.loadPCF(path)

L = analyzer.sim.L
V = (L*L*L)#/(2*2*2)
M = analyzer.sim.M
#num_snaps = 1000
#num_snaps_analyzed = int(0.75*1000)
norm_factor = V/(4*numpy.pi*M*M)
print((M, L, V, num_snaps_analyzed))
rho_avg = M/V
print("rho_avg=", rho_avg)

# Normalize
#print(y)
dr = r[1] - r[0]
y *= norm_factor * rho_avg / 2.45 #/num_snaps_analyzed
y[1:] /= (r[1:]*r[1:])
print(y.shape)
print("Mean:", dr*np.average(y))

# Plot the result
fig = plt.figure()#figsize=(12.8,9.6), dpi=200,)
plt.plot(r, y, 'r-x', label="density")
plt.plot(r,rho_avg*numpy.ones_like(r), 'b-', label='rho_avg')
plt.title("Volumetric density")
plt.xlabel("r... distance to particle in origin")
plt.ylabel("Volumetric density")
plt.xlim((0,3.1))
plt.legend()
plt.grid()
figpath = "snapshots/vol_dens_laura.png"
plt.savefig(figpath)
print(f"Saved plot in <{figpath}")