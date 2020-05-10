import jax.numpy as np
from jax import grad, jit
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from simulation import Simulation_box, Simulation_Analyzer, Epot_lj, grad_Epot, Verlet
# https://homepage.univie.ac.at/Franz.Vesely/simsp/dx/node17.html
parser = ArgumentParser(description='plots a pcf.txt file')
parser.add_argument('path', metavar='trajectory.xyz',type=str, help='a path to a trajectory file (.xyz);')
parser.add_argument('num', metavar='num',type=str, help='number of snaps analyzed --> take average')

# Parse arguments
args = parser.parse_args()
print(args)
path = args.path
num_snaps_analyzed = int(args.num)

analyzer = Simulation_Analyzer("snapshots/trajectory_dt0,03_p.xyz")
r,y = analyzer.loadPCF(path)

L = analyzer.sim.L
V = (L*L*L)#/(2*2*2)
#V = 1/6*numpy.pi*L*L*L
M = analyzer.sim.M
print("num snaps:", num_snaps_analyzed)
#num_snaps = 1000
#num_snaps_analyzed = int(0.75*1000)
#norm_factor = V/(4*numpy.pi*M*(M-1))/ numpy.sqrt(num_snaps_analyzed )
num_samples = y.size
dr = r[1] - r[0]
rho_avg = 0.8
norm_factor = V*rho_avg /(4*numpy.pi*M*(M-1)) / 2.2
print((M, L, V, num_snaps_analyzed))
print("rho_avg=", rho_avg)

# Normalize
#print(y)
print("dr:", dr)
#y = y * norm_factor # * rho_avg / 5#/ 2.45 #/num_snaps_analyzed
y[1:] = y[1:] / (np.power(r[1:],2)) * norm_factor
print(y.shape)
#print("Mean:", np.mean(y))

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
figpath = "snapshots/vol_dens_dt0,03p.png"
plt.savefig(figpath)
print(f"Saved plot in <{figpath}")