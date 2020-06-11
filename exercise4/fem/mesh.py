import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self,elements,numberElementsX,numberElementsY,L,coords):
        self.elements = elements
        self.numberNodes = (numberElementsX+1)*(numberElementsY+1)
        self.numberElementsX = numberElementsX
        self.numberElementsY = numberElementsY
        self.L = L
        self.coords = coords
        
    def drawMesh(self,number):
        for e in self.elements:
            self.drawElement(e,number)
        
    def drawElement(self,e,number):
        plt.plot([e.node1[0],e.node2[0]], [e.node1[1],e.node2[1]],'black')
        plt.plot([e.node2[0],e.node3[0]], [e.node2[1],e.node3[1]],'black')
        plt.plot([e.node3[0],e.node1[0]], [e.node3[1],e.node1[1]],'black')
        if(number):
            plt.text(e.center[0],e.center[1],e.ID)
        
    def assembleH(self):
        H = np.zeros((self.numberNodes,self.numberNodes))
        for e in self.elements:
            for m in range(3):
                for n in range(3):
                    H[e.ci[m]-1,e.ci[n]-1] += e.H[m,n]
        return H           
    
    def storeT(self,T):
        self.T = T
        
    def plotTemperatureField(self,numbers=False):
        fig,ax=plt.subplots(1,1)
        fig.set_size_inches(18.5, 10.5, forward=True)
        self.drawMesh(numbers)
        ax.set_title('Temperature field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        cp = plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet')
        fig.colorbar(cp)

        
    def calculateTempGradient(self):
        for e in self.elements:
            Ttemp = self.T[e.nodes-1]
            e.tempGrad = 1/(2*e.area) * np.array([[e.b1,e.b2,e.b3],[e.c1,e.c2,e.c3]]).dot(Ttemp)
            
    def calculateFlux(self):
        for e in self.elements:
            e.flux = np.array([e.tempGrad[0]*(-e.k), e.tempGrad[1]*(-e.k)])
            
    def plotTemperatureGradient(self):
        fig,ax=plt.subplots(1,1)
        self.drawMesh(False)
        plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet',alpha=0.2)
        x = []
        y = []
        u = []
        v = []
        M = []
        for e in self.elements:
            x.append(e.center[0])
            y.append(e.center[1])
            u.append(e.tempGrad[0])
            v.append(e.tempGrad[1])
            M.append(np.sqrt(e.tempGrad[0]*e.tempGrad[0]+e.tempGrad[1]*e.tempGrad[1]))
        image = plt.quiver(x,y,u,v,M,cmap=plt.cm.jet)
        plt.colorbar(image, cmap=plt.cm.jet)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_title('Temperature Gradient')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
    def plotFlux(self):
        fig,ax=plt.subplots(1,1)
        self.drawMesh(False)
        plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet',alpha=0.2)
        x = []
        y = []
        u = []
        v = []
        M = []
        for e in self.elements:
            x.append(e.center[0])
            y.append(e.center[1])
            u.append(e.flux[0])
            v.append(e.flux[1])
            M.append(np.sqrt(e.flux[0]*e.flux[0]+e.flux[1]*e.flux[1]))
        image = plt.quiver(x,y,u,v,M,cmap=plt.cm.jet)
        plt.colorbar(image, cmap=plt.cm.jet)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_title('Flux')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
    