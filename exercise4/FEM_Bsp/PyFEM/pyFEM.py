import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
	# right-hand side function for FEM
	return int(x[0]>0.5)*int(x[1]>0.5)-int(x[0]<0.5)*int(x[1]<0.5)

def phi(x):
	# Neumann data function for FEM
	return 0

class Element:
	def __init__(self,nodes,bdryedges):
		self.nodes = nodes		# nodes of the triangle [1,2,3]
		self.bdryedges = bdryedges	# boolean array which specifies if the edges [1,2], [2,3], [3,1] are on the boundary
	def getArea(self, coordinates):
		# compute area of element
		B = np.vstack(( coordinates[self.nodes[1],:]-coordinates[self.nodes[0],:],coordinates[self.nodes[2],:]-coordinates[self.nodes[0],:]))
		return 0.5*np.linalg.det(B)
	def getLocalGradients(self, coordinates):
		# compute gradients of local hat functions
		M = np.hstack((np.ones((3,1)), coordinates[self.nodes,:])).T
		rhs = np.array([[0,0],[1,0],[0,1]])
		return np.linalg.solve(M,rhs)
	

class Mesh:
	def __init__(self,coordinates,elements):
		self.coordinates = coordinates
		self.elements = elements
		self.nE = len(elements)			#number of elements
		self.nC = coordinates.shape[0]		#number of nodes/coordinates
	def getBoundaryNodes(self):
		# returns all nodes which lie on the boundary
		bdrynodes =[]
		for element in self.elements:
			if element.bdryedges[0]==True:
				bdrynodes.extend([element.nodes[0],element.nodes[1]])
			if element.bdryedges[1]==True:
				bdrynodes.extend([element.nodes[1],element.nodes[2]])
			if element.bdryedges[2]==True:
				bdrynodes.extend([element.nodes[2],element.nodes[0]])
		return np.unique(np.array(bdrynodes))

	def getInteriorNodes(self):
		# returns all interior nodes
		return np.setdiff1d(np.arange(self.nC),self.getBoundaryNodes())

	def getElementsArray(self):
		# returns a Matlab style element array (for plotting)
		tri=np.zeros((self.nE,3))
		for idx,element in enumerate(self.elements):
			tri[idx,:]=element.nodes
		return tri
	def plot(self,val = 0):
		# plots the mesh with optional function values in val
		fig = plt.figure(num=1, clear=True)
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		if all(val==0):
			val= np.zeros((self.nC,))
		else:
			val=val.reshape((self.nC,))
		ax.plot_trisurf(self.coordinates[:,0],self.coordinates[:,1],self.getElementsArray(),val)
		ax.set(xlabel='x', ylabel='y', zlabel='z')
		plt.show()
	

def buildStiffnessMatrix(mesh):
	# builds the Laplace stiffness matrix for P1 FEM
	A = np.zeros((mesh.nC,mesh.nC))
	for element in mesh.elements:
		nodes = element.nodes		
		grad = element.getLocalGradients(mesh.coordinates)
		localA = grad @ grad.T
		A[np.ix_(nodes,nodes)] += localA*element.getArea(mesh.coordinates)
	return A

def buildDirichletRHS(mesh):
	# builds the load vector for the Dirichlet Problem
	b = np.zeros((mesh.nC,1))
	for element in mesh.elements:
		nodes = element.nodes
		midpoint = (mesh.coordinates[nodes[0],:] + mesh.coordinates[nodes[1],:] + mesh.coordinates[nodes[2],:])/3		
		b[nodes] += 1./3.*element.getArea(mesh.coordinates)*f(midpoint)
	return b

def buildNeumannRHS(mesh):
	# builds the load vector for the Neumann Problem
	b = np.zeros((mesh.nC,1))
	for element in mesh.elements:
		if(any(element.bdryedges)):
			nodes = element.nodes
			edges = np.array([ [nodes[0],nodes[1]], [nodes[1],nodes[2]], [nodes[2],nodes[0]] ])
			bdryedges = edges[element.bdryedges,:]			
			for bdryedge in bdryedges:
				midpoint = (mesh.coordinates[bdryedge[0],:] + mesh.coordinates[bdryedge[1],:])/2
				length = np.sum((mesh.coordinates[bdryedge[0],:] - mesh.coordinates[bdryedge[1],:])**2)**(0.5)
				b[bdryedge] += 0.5*length*phi(midpoint)				
	return b+buildDirichletRHS(mesh)

def buildMeanVector(mesh):
	# builds the vector which contains the integral mean of hat functions
	c = np.zeros((mesh.nC,1))
	for element in mesh.elements:	
		nodes = element.nodes				
		c[nodes] += 1./3.*element.getArea(mesh.coordinates)
	return c


def uniformRefinement(mesh):
	# uniformly refines the mesh by RED-refinement
	edges = []
	element2edges = np.zeros((mesh.nE,3))
	counter = 0
	for idx,element in enumerate(mesh.elements):
		edges.append(sorted([element.nodes[0],element.nodes[1]]))
		edges.append(sorted([element.nodes[1],element.nodes[2]]))
		edges.append(sorted([element.nodes[2],element.nodes[0]]))
		element2edges[idx,:] = np.array([counter, counter +1, counter +2])
		counter += 3

	element2edges=element2edges.astype(int)
	edges, idx,iidx = np.unique(np.array(edges),axis=0, return_index=True, return_inverse=True)	
	element2edges=iidx[element2edges]
	newcoordinates = np.vstack((mesh.coordinates,0.5*(mesh.coordinates[edges[:,0],:] + mesh.coordinates[edges[:,1],:])))
	
	newelements = []	
	for idx,element in enumerate(mesh.elements):
		nodes = element.nodes
		bdryedges = element.bdryedges
		newelements.append(Element([nodes[0],element2edges[idx,0]+mesh.nC,element2edges[idx,2]+mesh.nC], [bdryedges[0],False,bdryedges[2]]))
		newelements.append(Element([element2edges[idx,0]+mesh.nC,nodes[1],element2edges[idx,1]+mesh.nC], [bdryedges[0],bdryedges[1],False]))
		newelements.append(Element([element2edges[idx,2]+mesh.nC,element2edges[idx,1]+mesh.nC,nodes[2]], [False,bdryedges[1],bdryedges[2]]))
		newelements.append(Element([element2edges[idx,2]+mesh.nC,element2edges[idx,0]+mesh.nC,element2edges[idx,1]+mesh.nC], [False,False,False]))
	
	
	newmesh = Mesh(newcoordinates,newelements)
	return newmesh
	
	
		
	
	


