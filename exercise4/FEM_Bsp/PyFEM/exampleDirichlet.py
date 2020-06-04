from pyFEM import *


# specify a coarse grid
elements =  [Element([3, 0, 1], [False, True, True]), Element([0, 3, 2],[False, True, True])]
coordinates = np.array([[0,0],[1,0],[0,1],[1,1]])

# build the mesh
mesh = Mesh(coordinates,elements)

# refine uniformly a couple of times
for i in range(4):
	mesh = uniformRefinement(mesh)


## Dirichlet Problem
# find interior nodes
intNodes = mesh.getInteriorNodes()

# build stiffness matrices and remove boundary variables
A = buildStiffnessMatrix(mesh)
A = A[np.ix_(intNodes,intNodes)]

# build load vector and remove boundary variables
b = buildNeumannRHS(mesh)
b = b[intNodes]

# solve the system and reinsert zero boundary values
xint = np.linalg.solve(A,b)
x = np.zeros((mesh.nC,1))
x[intNodes] = xint 

# plot solution
mesh.plot(x)


