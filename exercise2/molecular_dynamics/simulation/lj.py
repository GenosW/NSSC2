#!D:\Studium\Code\NSSC\NSSC2\exercise2\molecular_dynamics\md\Scripts\python.exe
# import numpy as np
# from numpy.random import default_rng
import numpy as np

class Simulation_box:
    name = 'Simulation box'
    sig = 1/np.power(2, 1/6)

    def __init__(self, M:int =3, L:float =1, Sigma:float = 0.5, description:str ='some simulation', path:str =''):
        if path:
            Lsnap, Msnap, positions, velocities = self.loadSnapshot(path)
            self.M = Msnap
            self.L = Lsnap
            self.positions =  positions # like np.array((3,M))
            self.velocities = velocities
        else:
            self.M = M
            self.L = L
            self.positions =  self.generateInitialPositions(L, M) # like np.array((3,M))
            self.velocities =  self.generateInitialVelocities(M, Sigma)
        self.description = description

    @staticmethod
    def generateInitialPositions(L, M):
        return L*np.random.rand(3,M)

    @staticmethod
    def generateInitialVelocities(M, Sigma):
        mean = np.zeros(3)
        cov = Sigma * np.diag(mean, k=0)
        return np.random.multivariate_normal(mean,cov)

    def Vlj(self, r):
        return 4*[np.power(self.sig/r, 12) - np.power(self.sig/r, 6)]

    def Epot(self, molecules):
        dist_list = np.array([[xi.getDist(xj) for xj in molecules[i+1:]] for i, xi in enumerate(molecules)]).flatten()
        energy = np.sum(self.Vlj(dist_list))
        return energy

    def Verlet(self, molecules):
        pass

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
            for i in list(range(M)):
                row = f.readline().split()
                (x,y,z,vx,vy,vz) = row
                print("i=",i,"-", x,y,z,vx,vy,vz)
                pos.append([x,y,z])
                vel.append([vx,vy,vz])
        return L, M, pos, vel

    def saveSnapshot(self, path):
        pass

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
        return 0 # smth 


class TrajectoryOld: # Orts- + Bewegungsdaten von einem Teilchen
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def getPos(self):
        return np.asarray([self.x, self.y, self.z], dtype=float)

    def getDist(self, pos): # pos should be either list or np.array like [x, y, z]
        myPos = np.asarray([self.x, self.y, self.z], dtype=float)
        dist = np.linarlg.norm(myPos - pos)
        return dist

    def getVel(self):
        return np.asarray([self.x, self.y, self.z], dtype=float)

    def getAbsVel(self):
        v = np.linarlg.norm((self.getVel()))
        return v

    def __repr__(self): # print(TrajectoryInstance) --> {'pos': Position as np.array, 'vel': Velocity as np.array}
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
