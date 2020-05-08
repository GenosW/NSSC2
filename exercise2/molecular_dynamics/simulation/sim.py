# import numpy as np
# from numpy.random import default_rng
import jax.numpy as np
from jax import grad, jit
import numpy
import scipy


def Epot_lj(positions, L:float, M:int):
    """Potential energy for Lennard-Jones potential in reduced units.
        In this system of units, epsilon=1 and sigma=2**(-1. / 6.). """
    # print(positions.shape)
    if positions.ndim == 1 and positions.size == M*3:
        positions = positions.reshape((M,3))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an Mx3 array or a 1D array that can be reshaped to Mx3!")
    # Compute all squared distances between pairs without iterating.
    delta = positions[:, np.newaxis, :] - positions
    delta = delta - L*np.around(delta/L, decimals=0)
    # print("delta")
    # print(delta)
    r2 = (delta * delta).sum(axis=2)
    # print("r2")
    # print(r2)
    # Take only the upper triangle (combinations of two atoms).
    indices = np.triu_indices(r2.shape[0], k=1)
    rm2 = 1. / r2[indices]
    # Compute the potental energy recycling as many calculations as possible.
    rm6 = rm2 * rm2 * rm2
    rm12 = rm6 * rm6
    return (rm12 - 2.*rm6).sum()

grad_Epot = jit(grad(Epot_lj), static_argnums=(1, 2))

def Verlet(x, v, dt, M, L):
    f = -grad_Epot(x.ravel(), L, M).reshape(M,3)
    x_new = numpy.zeros((M,3))
    v_new = numpy.zeros((M,3))
    for i in range(M):
        x_new[i,:] = x[i,:] + v[i,:]*dt + 0.5*f[i,:]*dt*dt
    f_new = -grad_Epot(x_new.ravel(), L, M).reshape(M,3)
    for i in range(M):
        v_new[i,:] = v[i,:] + 0.5*f[i,:]*dt + 0.5*f_new[i,:]*dt
    
    return x_new, v_new


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
            self.M, self.L, self.positions, self.velocities, descSnap = self.loadSnapshot(path)
            if Name:
                self.description = Name + ": " + descSnap
            else:
                self.description = descSnap
        else:
            self.M = M
            self.L = L
            self.positions = self.generateInitialPositions(
                L, M)  # like np.array((3,M))
            self.velocities = self.generateInitialVelocities(M, Sigma)
            if Name:
                self.description = Name + ": " + description
            else:
                self.description = description
        self.description.strip("\n")

    @staticmethod
    def generateInitialPositions(L, M):
        return L * numpy.random.rand(M, 3)

    @staticmethod
    def generateInitialVelocities(M, Sigma):
        mean = np.ones(3)
        cov = Sigma * np.diag(mean, k=0)
        return numpy.random.multivariate_normal(0 * mean, cov,
                                                size=M)  #.transpose()
    @staticmethod
    def loadSnapshot(path, offset=0):
        pos = []
        vel = []
        with open(path, 'r') as f:
            for _ in range(offset):
                line = f.readline()
                if not line:
                    return False
            row = f.readline().split()
            if not row:
                return False
            #print("Header detected")
            #print(row)
            M = int(row[0])
            description = f.readline().strip("\n")
            L = float(f.readline().split()[0])
            # print("M: ", M)
            # print("desc: ", description)
            # print("L: ", L)
            for _ in range(M):
                row = f.readline().split()
                row = [float(z) for z in row]
                (x, y, z, vx, vy, vz) = row
                #print("i=",i,"-", x,y,z,vx,vy,vz)
                pos.append([x, y, z])
                vel.append([vx, vy, vz])
        return M, L, np.asarray(pos), np.asarray(vel), description

    def loadSnapshotIntoBox(self, path, offset=0):
        ret = self.loadSnapshot(path, offset=offset)
        if not ret:
            return False
        (self.M, self.L, self.positions, self.velocities, self.description) = ret
        return True


    @staticmethod
    def getNumLines(path):
        with open(path, 'r') as file:
            num_lines = sum(1 for line in file)
            #num_lines = sum(1 for line in file if line.rstrip()) # alternative: strip empty lines
        return num_lines
    
    def saveSnapshot(self, path, mode):
        with open(path, mode) as f:
            header = [
                str(self.M) + "\n", self.description + "\n",
                str(self.L) + "\n"
            ]
            f.writelines(header)
            output = np.concatenate((self.positions, self.velocities), axis=1)
            numpy.savetxt(f, output, fmt='%.4e', delimiter=' ')

    @staticmethod
    def saveSnapshotStatic(positions, velocities, M, L, description, path, mode):
        with open(path, mode) as f:
            header = [
                str(M) + "\n", description + "\n",
                str(L) + "\n"
            ]
            f.writelines(header)
            output = np.concatenate((positions, velocities), axis=1)
            numpy.savetxt(f, output, fmt='%.4e', delimiter=' ')

    def moveToMinimumEnergy(self):
        result = scipy.optimize.minimize(Epot_lj, self.positions.ravel(), jac=grad_Epot, method='CG', args=(self.L, self.M), options={'gtol': 5e-3})
        newPositions = result.x
        print("Optimizations successful: ", result.success)
        print("Message: ", result.message)
        self.positions = newPositions.reshape(self.M,3)
        print("Sanity check: ", self.sanity_check(), "...should be [0, 0, 0].")

    def sanity_check(self):
        return np.around(grad_Epot(self.positions.ravel(), self.L, self.M).reshape(self.M,3).sum(axis=0), decimals=4) # (sum(fx_i), sum(fy_i), sum(fz_i))

    def average_velocity(self):
        return self.velocities.sum(axis=0)/self.M

    def Ekin(self):
        return 0.5*np.power(self.velocities, 2).sum(axis=0)

    def toCOM(self):
        self.velocities = self.velocities - self.average_velocity()

    @staticmethod
    def enforceMI(positions, L):
        return positions - L * np.around(positions/L, decimals=0)


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
