import numpy as np
import matplotlib.pyplot as plt
import math
from .element import Element
from .mesh import Mesh

# elements to be modified as given in exercise handout
elementsDefault = [61, 62, 63, 64, 65, 66, 67, 68, 79, 80, 81, 82, 83, 84, 85, 97, 98, 99, 100, 101, 115, 116, 117, 118]

class Triangulation:
    def __init__(self,L,numberElementsX,numberElementsY,k,h,mode = 0,factorV1 = 1, c=1):
        """
        L... length of square domain
        numberElementsX / Y... how many elements in x and y direction
        k... heat coefficient
        h... height of the domain
        mode... 0=base calculation, 1=Variation 1 etc
        factorV1... how much of its original length is upper edge of domain, used for trapezoidal mesh, can be used in any mode though
        c... modifies k of elements in elements_to_be_modified with k=k*c when used in mode 4
        """

        self.L = L
        self.numberElementsX = numberElementsX
        self.numberElementsY = numberElementsY
        self.k = k
        self.h = h
        self.mode = mode
        self.factorV1 = factorV1
        self.c = c
        
    def rotate(self,x,y,xo,yo,theta): 
        #rotate x,y around xo,yo by theta (rad), from stack overflow
        xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
        yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
        return [xr,yr]
        
    def createTriangulation(self, elements_to_be_modified=elementsDefault):
        if (self.mode in [0,1,2,3,4]):
            x = self.L/self.numberElementsX
            y = self.L/self.numberElementsY
            elementlist = []
            f = (1-self.factorV1)/self.numberElementsY
            coords = np.ones((2,(self.numberElementsX+1)*(self.numberElementsY+1)))
            for j in np.arange(self.numberElementsY+1):
                for i in np.arange(self.numberElementsX+1):
                    coords[0,j*(self.numberElementsX+1)+i] = i*x*(1-j*f)
                    if (self.mode == 2):
                        B = j*y/(2*self.L)
                        coords[0,j*(self.numberElementsX+1)+i] = coords[0,j*(self.numberElementsX+1)+i]*(B/self.L*coords[0,j*(self.numberElementsX+1)+i]-B+1)
                    coords[1,j*(self.numberElementsX+1)+i] = j*y
                    if (self.mode == 3):
                        coords[0,j*(self.numberElementsX+1)+i], coords[1,j*(self.numberElementsX+1)+i] = self.rotate(i*x*(1-j*f), 0, self.L*2, 0, -math.pi/4/self.numberElementsY*j)
            for j in np.arange(self.numberElementsY):
                for i in np.arange(self.numberElementsX):
                    nodesLower = np.array([j*(self.numberElementsX+1)+i+1,j*(self.numberElementsX+1)+i+2,(j+1)*(self.numberElementsX+1)+i+1])
                    coordLower = np.array([[coords[0,nodesLower[0]-1],coords[1,nodesLower[0]-1]], [coords[0,nodesLower[1]-1],coords[1,nodesLower[1]-1]], [coords[0,nodesLower[2]-1],coords[1,nodesLower[2]-1]]])  
                    idLower = j*self.numberElementsX*2+i*2+1
                    K = self.k
                    if (self.mode == 4 and idLower in elements_to_be_modified):
                        K = K*self.c
                    eLower = Element(coordLower, nodesLower, idLower, K, self.h)

                    nodesUpper = np.array([(j+1)*(self.numberElementsX+1)+i+1, j*(self.numberElementsX+1)+i+2, (j+1)*(self.numberElementsX+1)+i+2])
                    coordUpper = np.array([[coords[0,nodesUpper[0]-1],coords[1,nodesUpper[0]-1]], [coords[0,nodesUpper[1]-1],coords[1,nodesUpper[1]-1]], [coords[0,nodesUpper[2]-1],coords[1,nodesUpper[2]-1]]])
                    idUpper = j*self.numberElementsX*2+i*2+2
                    K = self.k
                    if (self.mode == 4 and idUpper in elements_to_be_modified):
                        K = K*self.c
                    eUpper = Element(coordUpper, nodesUpper, idUpper, K, self.h)
                    

                    elementlist.append(eLower)
                    elementlist.append(eUpper)

            M = Mesh(elementlist,self.numberElementsX,self.numberElementsY,self.L,coords)
            
        return M