import jax.numpy as np
import numpy
from .sim import Simulation_box, Epot_lj, grad_Epot

class Simulation_Analyzer:

    def __init__(self, path):
        self.path = path
        self.Epots = []
        self.Ekins = []
        self.Etots = []
        self.sim = Simulation_box(path=path)
        
    def calculateEnergies(self):
        Epot = Epot_lj(self.sim.positions, self.sim.L, self.sim.M)
        Ekin = self.sim.Ekin()
        Etot = Epot + Ekin
        self.Epots.append(Epot)
        self.Ekins.append(Ekin)
        self.Etots.append(Etot)

    def analyze(self, path=None, num_samples=50, start=0., stop=1):
        if not path:
            path = self.path
            
        # Volumen of the simulation box
        V = self.sim.L * self.sim.L * self.sim.L
        # Lines in file per snapshot
        lines_per_snap = self.sim.M + 3
        # Number of lines in the file
        lines_in_file = self.sim.getNumLines(path)

        print("Lines in file:", lines_in_file)
        print("Lines per snap:", lines_per_snap)
        # Number of snapshots in the file (=N...number of steps in evolve.py)
        num_snaps = lines_in_file/lines_per_snap
        print("Number of snaps: ", num_snaps)
        if isinstance(num_snaps, float):
            assert(num_snaps.is_integer())
            num_snaps = int(num_snaps)
        elif isinstance(num_snaps, int):
            pass
        else:
            raise TypeError("There is something wrong with the input file...cannot analyze!")

        # <num_samples> samples evenly spaced on the intervall [0, L/2]
        samples, dr = np.linspace(0, self.sim.L/2, num=num_samples, retstep=True) 
        pcf = numpy.zeros_like(samples)

        start_offset = int(start*num_snaps*lines_per_snap) # end of first 25% of frames
        stop_offset = stop*num_snaps*lines_per_snap # Frame number to stop at 
        offset = 0
        j = 0
        # For the first 25%, we only calculate energies
        while self.sim.loadSnapshotIntoBox(path, offset=offset):
            self.calculateEnergies()
            j += 1
            offset = j*lines_per_snap
            if offset >= start_offset: 
                break
        offset = start_offset
        i = 0
        print("offset:",offset)
        # For the remaining 75%, we calculate energies + the pair density function (not normalized yet)
        while self.sim.loadSnapshotIntoBox(path, offset=offset):
            self.calculateEnergies()
            delta = self.sim.positions[:, np.newaxis, :] - self.sim.positions
            delta = delta - self.sim.L*np.around(delta/self.sim.L, decimals=0)

            bins = numpy.trunc( numpy.sqrt((delta*delta).sum(axis=2)) /dr )
            bins = bins.astype(int) # round to nearest int
            indices = numpy.triu_indices(bins.shape[0], k=1) # only want upper triangle --> no double counting
            for k in bins[indices]:
                if k >= num_samples:
                    print(k)
                    #k_tmp = 1 - self.sim.L/2
                else:
                    pcf[k] += 1
            # pcf[bins[indices]] += 1
            i += 1
            offset = start_offset + i*lines_per_snap
            if offset >= stop_offset:
                break
        print("Snaps for used for PCF:",i)
        # normalize 
        #pcf *= V/(num_samples*num_samples)
        return samples, pcf, (i, self.sim.M, self.sim.L)

    def savePCF(self, r, pcf, path):
        out = numpy.stack((r, pcf), axis=1)
        with open(path, 'w') as f:
            numpy.savetxt(f, out, fmt='%.4e', delimiter=' ')

    def loadPCF(self, path):
        with open(path, 'r') as f:
            in_arr = numpy.loadtxt(f)
            #print(in_arr)
            assert(in_arr.shape[1] == 2)
            return in_arr[:,0], in_arr[:,1]

    def saveEnergies(self, path):
        out = numpy.stack((np.array(self.Epots), np.array(self.Ekins), np.array(self.Etots)), axis=1)
        with open(path, 'w') as f:
            numpy.savetxt(f, out, fmt='%.4e', delimiter=' ')

    def loadEnergies(self, path):
        with open(path, 'r') as f:
            in_arr = numpy.loadtxt(f)
            #print(in_arr)
            assert(in_arr.shape[1] == 3)
            return in_arr[:,0], in_arr[:,1], in_arr[:,2]

