#!D:\Studium\Code\NSSC\NSSC2\exercise2\molecular_dynamics\md\Scripts\python.exe
# import numpy as np
# from numpy.random import default_rng
import jax.numpy as np
from jax import grad, jit
import numpy


def Epot_lj(positions):
    """Potential energy for Lennard-Jones potential in reduced units.
        In this system of units, epsilon=1 and sigma=2**(-1. / 6.). """
    print(positions.shape)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an Mx3 array")
    # Compute all squared distances between pairs without iterating.
    delta = positions[:, np.newaxis, :] - positions
    r2 = (delta * delta).sum(axis=2)
    # Take only the upper triangle (combinations of two atoms).
    indices = np.triu_indices(r2.shape[0], k=1)
    rm2 = 1. / r2[indices]
    # Compute the potental energy recycling as many calculations as possible.
    rm6 = rm2 * rm2 * rm2
    rm12 = rm6 * rm6
    return (2. * rm12 - rm6).sum()


grad_Epot = jit(grad(Epot_lj))


def Verlet(self, x, v, f, dt):
    #forces = -grad_Epot(x)
    new_x = x + (v + dt / 2 * f) * dt
    new_f = -grad_Epot(new_x)
    new_v = v + (f + new_f) * dt / 2
    return new_x, new_v, new_f


class Simulation_box:
    name = 'Simulation box'
    sig = 1 / np.power(2, 1 / 6)

    def __init__(self,
                 M: int = 3,
                 L: float = 1,
                 Sigma: float = np.power(0.5, 1 / 6),
                 description: str = 'some simulation',
                 path: str = '',
                 Name=""):
        if path:
            Lsnap, Msnap, positions, velocities = self.loadSnapshot(path)
            self.M = Msnap
            self.L = Lsnap
            self.positions = positions  # like np.array((M,3))
            self.velocities = velocities
        else:
            self.M = M
            self.L = L
            self.positions = self.generateInitialPositions(
                L, M)  # like np.array((3,M))
            self.velocities = self.generateInitialVelocities(M, Sigma)
            print("positions:", self.positions)
            print("velocities:", self.velocities)
        if Name:
            self.description = Name + ": " + description
        else:
            self.description = description

    @staticmethod
    def generateInitialPositions(L, M):
        return L * numpy.random.rand(M, 3)

    @staticmethod
    def generateInitialVelocities(M, Sigma):
        mean = np.ones(3)
        cov = Sigma * np.diag(mean, k=0)
        return numpy.random.multivariate_normal(0 * mean, cov,
                                                size=M)  #.transpose()

    def loadSnapshot(self, path, M=None):
        pos = []
        vel = []
        with open(path, 'r') as f:
            row = f.readline().split()
            if len(row) == 1 and not M:
                print("Header detected")
                M = int(row[0])
                description = f.readline()
                L = float(f.readline().split()[0])
                print("M: ", M)
                print("desc: ", description)
                print("L: ", L)
            for _ in list(range(M)):
                row = f.readline().split()
                row = [float(z) for z in row]
                (x, y, z, vx, vy, vz) = row
                #print("i=",i,"-", x,y,z,vx,vy,vz)
                pos.append([x, y, z])
                vel.append([vx, vy, vz])
        return L, M, pos, vel

    def saveSnapshot(self, path, mode):
        with open(path, mode) as f:
            header = [
                str(self.M) + "\n", self.description + "\n",
                str(self.L) + "\n"
            ]
            f.writelines(header)
            output = np.concatenate((self.positions, self.velocities), axis=1)
            numpy.savetxt(f, output, fmt='%.4e', delimiter=' ')


class Trajectory:
    def __init__(self, input_snap="traj_example.txt", dt=0.001, N=10):
        self.input = input_snap
        self.dt = 0.001
        self.N = 10
        SimBox.readSnapshot(input_snap)
        self.SimBox = 1
        self.trajectories = []

    def timeStep(self):
        # Do some Netwon stuff
        # Verlet algorithm
        return newPositions, newVelocities

    def iterate(self, N):
        for n in list(range(N)):
            newPos, newVel = self.timeStep()
            newBox = Simulation_Box(newPos, newVel)
            newBox.saveSnapshot()
            self.trajectories.append(newBox)
            # whatever else needds to be done
        return 0  # smth


class TrajectoryOld:  # Orts- + Bewegungsdaten von einem Teilchen
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def getPos(self):
        return np.asarray([self.x, self.y, self.z], dtype=float)

    def getDist(self,
                pos):  # pos should be either list or np.array like [x, y, z]
        myPos = np.asarray([self.x, self.y, self.z], dtype=float)
        dist = np.linarlg.norm(myPos - pos)
        return dist

    def getVel(self):
        return np.asarray([self.x, self.y, self.z], dtype=float)

    def getAbsVel(self):
        v = np.linarlg.norm((self.getVel()))
        return v

    def __repr__(
        self
    ):  # print(TrajectoryInstance) --> {'pos': Position as np.array, 'vel': Velocity as np.array}
        return str({'pos': self.getPos(), 'vel': self.getVel()})


'''
Line 1: Number of atoms (integer)
Line 2: Arbitrary comment/description (free format)
Line 3: Box side length (decimal number)
Line 4: x1 y1 z1 vx1 vy1 vz1
.
.
.
Line M+3: xM yM zM vxM vyM vzM
'''
