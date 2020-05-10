'''init_md.py [M] [L] [Sigma]
generates a ”relaxed” (i.e., low-energy) starting confguration for a molecular dynamics simulation.

Takes 3 command line arguments (callable as above):
    M... a number of particles;
    L... a side length for the simulation box; and
    Sigma... a standard deviation for the velocity distribution
'''
from argparse import ArgumentParser
from simulation import Simulation_box, Epot_lj, grad_Epot

# Command-line argument parser
parser = ArgumentParser(
    description=
    'generates a ”relaxed” (i.e., low-energy) starting confguration for a molecular dynamics simulation.'
)
parser.add_argument('M', metavar='M', type=int, help='a number of particles;')
parser.add_argument('L',
                    metavar='L',
                    type=float,
                    help='a side length for the simulation box;')
parser.add_argument('Sigma',
                    metavar='Sigma',
                    type=float,
                    help='a standard deviation for the velocity distribution')
parser.add_argument(
    '-name',
    metavar='name',
    type=str,
    help=
    'Name/ID of simulation that is, if given, prepended to the description.',
    default="")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    M = args.M
    L = args.L
    Sigma = args.Sigma
    name = args.name
    print("Initializing molecular dynamics simulation with:")
    print(f"M = {M}")
    print(f"L = {L}")
    print(f"Sigma = {Sigma}")
    if name:
        print(f"Name:{name}")

    snapshotDir = "snapshots/"
    saveFile = snapshotDir + "init.xyz"

    sim = Simulation_box(M, L, Sigma, Name=name)
    #sim.saveSnapshot(saveFile, mode="a")
    print("Epot1:", Epot_lj(sim.positions, L, M))
    #print("Epot2:",sim.Epot_Ch(sim.positions))
    print("#" * 40)
    print("Minimizing...")
    sim.moveToMinimumEnergy()
    print("CG done!")
    print("#" * 40)
    print("sim.positions.shape =", sim.positions.shape)
    print("Epot1:", Epot_lj(sim.positions, L, M))
    #print("Epot2:",sim.Epot_Ch(sim.positions))
    #print("Forces: ", -grad_Epot(sim.positions.ravel().reshape(M,3), L, M))
    print("Average velocity: ", sim.average_velocity())
    print("centering: ")
    sim.toCOM()
    sim.positions = sim.enforceMI()
    print("Average velocity now: ", sim.average_velocity())
    print("Epot1:", Epot_lj(sim.positions, L, M))
    sim.saveSnapshot(saveFile, mode="w")
