# import numpy as np
# from numpy.random import default_rng
import jax.numpy as np
from jax import grad, jit
import numpy
from scipy.optimize import minimize


def Epot_lj(positions, L: float, M: int):
    """Potential energy for Lennard-Jones potential in reduced units.
        In this system of units, epsilon=1 and sigma=2**(-1. / 6.). 
        
        The function accepts numpy arrays of shape (M, 3) [2D] or (M*3) [1D]."""
    if (positions.ndim != 2 or positions.shape[1] != 3) and not (
            positions.ndim == 1 and positions.size == M * 3):
        raise ValueError(
            "positions must be an Mx3 array or a 1D array that can be reshaped to Mx3!"
        )
    if positions.ndim == 1 and positions.size == M * 3:
        positions = positions.reshape((M, 3))  # Reshape to Mx3
    #sig = 1 / np.power(2, 1 / 6)
    sig = 1.

    # Compute all squared distances between pairs
    delta = positions[:, np.newaxis, :] - positions
    delta = delta - L * np.around(delta / L, decimals=0)
    r2 = (delta * delta).sum(axis=2)  # r^2 ...squared distances

    # Take only the upper triangle (combinations of two atoms).
    indices = np.triu_indices(r2.shape[0], k=1)
    rm2 = sig * sig / r2[indices]  # (sig/r)^2
    # Compute the potental energy recycling as many calculations as possible.
    rm6 = rm2 * rm2 * rm2  # (sig/r)^6
    rm12 = rm6 * rm6  # (sig/r)^12
    return (rm12 - 2. * rm6).sum()


# Define the gradient of Epot via jax automated differentiation and jit
grad_Epot = jit(grad(Epot_lj), static_argnums=(1, 2))


def Verlet(x, v, dt, M, L):
    """Integration of Newton's equations of motion via the Verlet algorithm.

        This implementation needs only a function to be defined for the gradient of Epot: grad_Epot
    """
    # Calculate forces for current iteration step
    f = -grad_Epot(x.ravel(), L, M).reshape(M, 3)

    # Calculate new positions
    x_new = x + (v + 0.5 * f * dt) * dt
    # Calculate new forces based on the new positions
    f_new = -grad_Epot(x_new.ravel(), L, M).reshape(M, 3)
    # Calculate new velocities based on average of old and new forces
    v_new = v + (f + f_new) * 0.5 * dt
    return x_new, v_new


class Simulation_box:
    name = 'Simulation box'

    def __init__(self,
                 M: int = 3,
                 L: float = 1,
                 Sigma: float = np.power(0.5, 1 / 6),
                 description: str = 'some simulation',
                 path: str = '',
                 Name=""):
        """Constructer for the Simulation_Box class.

        There are 2 modes for calling the constructor:
        1) If a path is provided, it is assumed to be a path to a file in .xyz format (or other text based) that contains a snapshot of a system, including the parameters M, L. The Simulation_Box instance is then initialized with the data from the snapshot.

        OR

        2) If no path is given (empty string), then a new Simulation_Box instance is created. The parameters M, L are taken from the call signature (either values are given or defaults as above). Positions and velocities generated randomly (see below).
        """
        if path:
            self.M, self.L, self.positions, self.velocities, descSnap = self.loadSnapshot(
                path)
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
        """Generate initial positions based on a uniform distribution"""
        return L * numpy.random.rand(M, 3)

    @staticmethod
    def generateInitialVelocities(M, Sigma):
        """Generate initial velocites based on a (3D) Gauß distribution with Mean = [0,0,0] and Variance = Sigma^2 in every component (Cov-Matrix diag matrix with Sigma^2 in diagonal)"""
        mean = np.ones(3)
        cov = numpy.sqrt(Sigma) * numpy.diag(mean, k=0)
        return numpy.random.multivariate_normal(0 * mean, cov, size=M)

    @staticmethod
    def loadSnapshot(path, offset=0):
        """Load a snapshot from a text file in .xyz-format.
        
        Input files should have the following format:
            Line 1: M = Number of atoms (integer)
            Line 2: Arbitrary comment/description (free format)
            Line 3: L = Box side length (decimal number)
            Line 4: x1 y1 z1 vx1 vy1 vz1
            .
            .
            .
            Line M+3: xM yM zM vxM vyM vzM
        """
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
            M = int(row[0])
            description = f.readline().strip("\n")
            L = float(f.readline().split()[0])
            for _ in range(M):
                row = f.readline().split()
                row = [float(z) for z in row]
                (x, y, z, vx, vy, vz) = row
                pos.append([x, y, z])
                vel.append([vx, vy, vz])
        return M, L, np.asarray(pos), np.asarray(vel), description

    def loadSnapshotIntoBox(self, path, offset=0):
        """Convenience function/method: Uses the above staticmethod to load a snapshot directly into the current class instance"""
        ret = self.loadSnapshot(path, offset=offset)
        if not ret:
            return False
        (self.M, self.L, self.positions, self.velocities,
         self.description) = ret
        return True

    @staticmethod
    def getNumLines(path):
        """Count number of lines in a file"""
        with open(path, 'r') as file:
            num_lines = sum(1 for line in file)
            #num_lines = sum(1 for line in file if line.rstrip()) # alternative: strip empty lines
        return num_lines

    def saveSnapshot(self, path="", mode="w"):
        """Save a snapshot (current positions and velocities in class instance) to a text file in .xyz-format.
        
        Output file have the following format:
            Line 1: M = Number of atoms (integer)
            Line 2: Arbitrary comment/description (free format)
            Line 3: L = Box side length (decimal number)
            Line 4: x1 y1 z1 vx1 vy1 vz1
            .
            .
            .
            Line M+3: xM yM zM vxM vyM vzM
        """
        with open(path, mode) as f:
            header = [
                str(self.M) + "\n", self.description + "\n",
                str(self.L) + "\n"
            ]
            f.writelines(header)
            output = np.concatenate((self.positions, self.velocities), axis=1)
            numpy.savetxt(f, output, fmt='%e', delimiter=' ')

    @staticmethod
    def saveSnapshotStatic(positions, velocities, M, L, description, path,
                           mode):
        """Staticmethod version of saveSnapshot"""
        with open(path, mode) as f:
            header = [str(M) + "\n", description + "\n", str(L) + "\n"]
            f.writelines(header)
            output = np.concatenate((positions, velocities), axis=1)
            numpy.savetxt(f, output, fmt='%e', delimiter=' ')

    def moveToMinimumEnergy(self):
        """Move the particles, whose positions and velocities are stored in the class instance, to a local energy minimum. A sanity check is performed after the minimization is complete by checking the (component wise) sum of all forces which should return [0,0,0] (sum of all forces in a closed system is zero == conservation of momentum).

        The method used for the minimizer is CG with gtol=1e-3*M (M...number of atoms). The minimizer is given the potential function Epot and its gradient grad_Epot.
        """
        result = minimize(Epot_lj,
                          self.positions.ravel(),
                          jac=grad_Epot,
                          method='CG',
                          args=(self.L, self.M),
                          options={'gtol': 1e-3 * self.M})
        newPositions = result.x
        print("Optimizations successful: ", result.success)
        print("Message: ", result.message)
        self.positions = newPositions.reshape(self.M, 3)
        print("Sanity check: ", self.sanity_check(), "...should be [0, 0, 0].")

    def sanity_check(self):
        """Sanity check performed after minimization... conservation of momentum"""
        return np.around(grad_Epot(self.positions.ravel(), self.L,
                                   self.M).reshape(self.M, 3).sum(axis=0),
                         decimals=6)  # [sum(fx_i), sum(fy_i), sum(fz_i)]

    def average_velocity(self):
        """Calculate average velocity of particles in the Simulation_Box"""
        return self.velocities.sum(axis=0) / self.M

    def Ekin(self):
        """Calculate the kinetic energy of particles in the Simulation_Box"""
        return 0.5 * np.power(self.velocities, 2).sum()

    def toCOM(self):
        """Subtract the average velocity from all velocities... to Center Of Mass"""
        self.velocities = self.velocities - self.average_velocity()

    def enforceMI(self):
        """Enforce the Minimum Image convention on all positions in the instance"""
        return self.positions - self.L * np.around(self.positions / self.L,
                                                   decimals=0)
