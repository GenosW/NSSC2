import jax.numpy as np
import numpy
from IPython import display 

arr = np.zeros((3,3))
print(arr)


path = 'snap.txt'
M = None
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
                row = [float(i) for i in row]
                (x,y,z,vx,vy,vz) = row
                print("i=",i,"-", x,y,z,vx,vy,vz)
                pos.append([x,y,z])
                vel.append([vx,vy,vz])
print(pos)
print(vel)
pos = np.asarray(pos)
vel = np.asarray(vel)

Sigma = np.power(0.5, 1/6)
print("Multivariate")
print(Sigma)
mean = np.ones(3)
cov = Sigma * np.diag(mean, k=0)

print("mean", mean)
print("cov", cov)
b = numpy.random.multivariate_normal(0*mean,cov, size=3)
print("new_v=",b)
a = np.concatenate((pos, vel), axis=1)
print(a)
with open('out.xyz', "w") as f:
        numpy.savetxt(f, a)
#with open(path, mode) as f: