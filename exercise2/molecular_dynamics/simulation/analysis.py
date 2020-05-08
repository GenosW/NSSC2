import jax.numpy as np
import numpy
from .sim import Simulation_box, Epot_lj, grad_Epot

class Simulation_Analyzer:

    def __init__(self, path):
        self.path = path
        self.Epots = []
        self.Ekins = []
        self.Etot = []
        self.sim = Simulation_box(path=path)
        #self.calculateEnergies()
        
    def calculateEnergies(self):
        Epot = Epot_lj(self.sim.positions, self.sim.L, self.sim.M)
        Ekin = self.sim.Ekin()
        Etot = Epot + Ekin
        self.Epots.append(Epot)
        self.Ekins.append(Ekin)
        self.Etot.append(Etot)
        print(f"Epot = {Epot}")
        print(f"Ekin = {Ekin}")

    def analyze(self, path=None, num_samples=50, start=0., stop=100):
        if not path:
            path = self.path
            
        V = self.sim.L * self.sim.L * self.sim.L
        lines_per_snap = self.sim.M + 3
        lines_in_file = self.sim.getNumLines(path)
        print("Lines in file:", lines_in_file)
        print("Lines per snap:", lines_per_snap)
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
        samples, dr = np.linspace(0, self.sim.L/2*np.sqrt(3), num=num_samples, retstep=True) 
        pcf = numpy.zeros_like(samples)

        offset = int(start*num_snaps*lines_per_snap)
        print("offset:",offset)
        i = 0
        while self.sim.loadSnapshotIntoBox(path, offset=offset):
            delta = self.sim.positions[:, np.newaxis, :] - self.sim.positions
            delta = delta - self.sim.L*np.around(delta/self.sim.L, decimals=0)
            bins = numpy.rint( numpy.sqrt((delta*delta).sum(axis=2)) /dr ) # round to nearest int
            indices = numpy.triu_indices(delta.shape[0], k=1) # only want upper triangle --> no double counting
            for k in bins[indices]:
                k_tmp = k
                if k_tmp >= num_samples-1:
                    print(k_tmp)
                    k_tmp = 49
                    #k_tmp = 1 - self.sim.L/2
                else:
                    pcf[int(k_tmp)] += 1
            i += 1
            offset = i*lines_per_snap
            if offset >= stop*num_snaps*lines_per_snap:
                break
        print(i)
        # normalize 
        pcf *= V/(num_samples*num_samples)
        return samples, pcf
