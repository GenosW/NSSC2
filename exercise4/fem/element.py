"""
This is the element module. The Element class is defined here.
"""
import numpy as np
import matplotlib.pyplot as plt

class Element:
    """
    This is an element for a FEM.
    """
    def __init__(self,coordinates,nodes,ID,k,h):
        """
        This is the constructor for element.
        coordinates... coordinates
        
        Hi from Peter!
        """
        self.ID = ID
        self.node1 = np.array([coordinates[0,0], coordinates[0,1], nodes[0]])
        self.node2 = np.array([coordinates[1,0], coordinates[1,1], nodes[1]])
        self.node3 = np.array([coordinates[2,0], coordinates[2,1], nodes[2]])
        self.nodes = nodes
        self.center = np.array([sum(coordinates[:,0])/3, sum(coordinates[:,1])/3])
        self.k = k
        self.tempGrad = 0
        self.flux = 0
        x1 = coordinates[0,0]
        x2 = coordinates[1,0]
        x3 = coordinates[2,0]
        y1 = coordinates[0,1]
        y2 = coordinates[1,1]
        y3 = coordinates[2,1]
        self.a1 = x2*y3 - x3*y2
        self.a2 = x3*y1 - x1*y3
        self.a3 = x1*y2 - x2*y1
        self.b1 = y2 - y3
        self.b2 = y3 - y1
        self.b3 = y1 - y2
        self.c1 = x3 - x2
        self.c2 = x1 - x3
        self.c3 = x2 - x1
        self.area = (x1*self.b1 + x2*self.b2 + x3*self.b3)/2
        self.ci = nodes  # coincidence matrix
        
        self.H = self.k*h/(4*self.area)*np.array([[self.b1*self.b1+self.c1*self.c1, self.b1*self.b2+self.c1*self.c2, self.b1*self.b3+self.c1*self.c3], \
                                            [self.b2*self.b1+self.c2*self.c1, self.b2*self.b2+self.c2*self.c2, self.b2*self.b3+self.c2*self.c3], \
                                            [self.b3*self.b1+self.c3*self.c1, self.b3*self.b2+self.c3*self.c2, self.b3*self.b3+self.c3*self.c3]])