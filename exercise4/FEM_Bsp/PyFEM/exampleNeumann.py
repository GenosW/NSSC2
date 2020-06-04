from pyFEM import *


# specify a coarse grid
elements =  [Element([3, 0, 1], [False, True, True]), Element([0, 3, 2],[False, True, True])]
coordinates = np.array([[0,0],[1,0],[0,1],[1,1]])

# build the mesh
mesh = Mesh(coordinates,elements)

# refine uniformly a couple of times
for i in range(4):
	mesh = uniformRefinement(mesh)


## Neumann Problem

# build stiffness matrix
A = buildStiffnessMatrix(mesh)

# build load vector and mean vector
b = buildNeumannRHS(mesh)
c = buildMeanVector(mesh)

# assemble saddle-point system

M = np.vstack((np.hstack((A,c)),np.hstack((c.T,np.zeros((1,1))))))
r = np.vstack((b,np.zeros((1,1))))

# solve the system and reinsert zero boundary values
xlam = np.linalg.solve(M,r)
x = xlam[0:mesh.nC]
lam = xlam[mesh.nC]

# plot solution
print(lam)
mesh.plot(x)


