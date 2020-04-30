import numpy as np
from numpy.random import default_rng


class Trajectory:
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


class Simulation_box:
    name = 'Simulation box'
    sig = 1/np.power(2, 1/6)
    
    def __init__(self, M:int =3, L:float =1, molecules:list =[], description:str ='some simulation'):
        self.M = M
        self.L = L
        if molecules and len(molecules)==M:
            self.molecules = molecules
        else:
            self.molecules = self.loadMoleculesDefault(M, L)
        self.description = description

    def loadMoleculesDefault(self, M, L):
        rands = np.random.uniform(0, L, (M, 6))
        molecs = [Trajectory(*row) for row in rands]
        return molecs

    def Vlj(self, r):
        return 4*[np.power(self.sig/r, 12) - np.power(self.sig/r, 6)]

    def Epot(self, molecules):
        dist_list = np.array([[xi.getDist(xj) for xj in molecules[i+1:]] for i, xi in enumerate(molecules)]).flatten()
        energy = np.sum(self.Vlj(dist_list))
        return energy

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