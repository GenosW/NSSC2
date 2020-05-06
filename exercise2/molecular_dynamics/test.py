import jax.numpy as np 

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
                (x,y,z,vx,vy,vz) = row
                print("i=",i,"-", x,y,z,vx,vy,vz)
                pos.append([x,y,z])
                vel.append([vx,vy,vz])
print(pos)
print(vel)
