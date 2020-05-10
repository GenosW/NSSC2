"""Module for a molecular dynamics simulation.

The simulation revolves around a Simulation_Box class, whose instances contain positions and velocities of particles in a box and allows for the saving of system snapshots in .xyz-format. Trajectory files in .xyz-format can be created aswell.

The potential governing the interaction between the particles is given via the function Epot_lj (Lennard-Jones-Potential) and its gradient is given by the function grad_Epot (obtained via AD by JAX).
The integration of Newton's equations of motion is done via the Verlet algorithm (function Verlet)

Lastly, the Simulation_Analyzer class provides an interface to perform analysis on the Trajectory files created by the Simulation_Box. It can calculate the Epot, Ekin and Etot per snapshot. It can also calculate an approximation of the volumetric density function based on the pair correlation function (PCF) by averaging over a number of snapshots in a trajectory file.
Both the energies and the PCF are saved in seperate files.
"""
from .sim import *
from .analysis import *