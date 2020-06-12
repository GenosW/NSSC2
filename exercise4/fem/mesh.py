import numpy as np
import matplotlib.pyplot as plt
import os


overdrawFactor = 0.05

class Mesh:
    def __init__(self,elements,numberElementsX,numberElementsY,L,coords, plotDir="./plots"):
        """
        constructor of mesh
        elements... list of elements which build up the mesh
        numberElementsX / Y... how many elements there are in x and y direction
        L... length of square domain
        coords... coordinates of nodes in ascending order, used for plotting
        """

        self.elements = elements
        self.numberNodes = (numberElementsX+1)*(numberElementsY+1)
        self.numberElementsX = numberElementsX
        self.numberElementsY = numberElementsY
        self.numberElementsTotal = len(elements) #numberElementsX*numberElementsY
        self.L = L
        self.coords = coords
        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)
            print(f"Created {plotDir}")
        self._plotDir = plotDir
        
    def drawMesh(self,number):
        for e in self.elements:
            self.drawElement(e,number)
        
    def drawElement(self,e,number):
        plt.plot([e.node1[0],e.node2[0]], [e.node1[1],e.node2[1]],'black')
        plt.plot([e.node2[0],e.node3[0]], [e.node2[1],e.node3[1]],'black')
        plt.plot([e.node3[0],e.node1[0]], [e.node3[1],e.node1[1]],'black')
        if(number):
            plt.text(e.center[0],e.center[1],e.ID, horizontalalignment='center', verticalalignment='center', fontsize=16)
        
    def assembleH(self):
        """
        assembles the global stiffness matrix
        """

        H = np.zeros((self.numberNodes,self.numberNodes))
        for e in self.elements:
            for m in range(3):
                for n in range(3):
                    H[e.ci[m]-1,e.ci[n]-1] += e.H[m,n]
        return H           
    
    def storeT(self,T):
        self.T = T
        
    def calculateTempGradient(self):
        for e in self.elements:
            Ttemp = self.T[e.nodes-1]
            e.tempGrad = 1/(2*e.area) * np.array([[e.b1,e.b2,e.b3],[e.c1,e.c2,e.c3]]).dot(Ttemp)
            
    def calculateFlux(self):
        for e in self.elements:
            e.flux = np.array([e.tempGrad[0]*(-e.k), e.tempGrad[1]*(-e.k)])
            
    def setPlotDir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            print(f"Created {path}")
        self._plotDir = path
        
    def plotTemperatureField(self, name, numbers=False, saveFig=True, barSpacing="uniform"):
        fig,ax=plt.subplots(1,1)
        fig.set_size_inches(18.5, 10.5, forward=True)
        self.drawMesh(numbers)
        ax.set_title('Temperature field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        cp = plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet')
        fig.colorbar(cp, label="Temperature [K]", spacing=barSpacing)
        if saveFig:
            saveName = os.path.join(self._plotDir, name + '_TemperatureField.png')
            plt.savefig(saveName)
            
    def plotTemperatureGradient(self,name, saveFig=True, barSpacing="uniform", scaleArrowColor:tuple=None):
        fig,ax=plt.subplots(1,1)
        self.drawMesh(False)
        plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet',alpha=0.2)
        
        x, y = np.zeros(self.numberElementsTotal), np.zeros(self.numberElementsTotal)
        u, v = np.zeros_like(x), np.zeros_like(y)
        for i, e in enumerate(self.elements):
            x[i], y[i] = e.center
            u[i], v[i] = e.tempGrad
        M = np.sqrt(u*u + v*v)
            
        image = plt.quiver(x,y,u,v,M,cmap=plt.cm.jet)
        
        if not scaleArrowColor:
            # Determine max/min arrow lengths
            minArrow, maxArrow = min(M), max(M)
            if maxArrow - minArrow < 1e-4*maxArrow:
                minArrow = int(minArrow*1e-3)*1e3
            print(f"Temperature gradient - color_scale: {minArrow} - {maxArrow}")
            plt.clim(minArrow,maxArrow)
        else:
            assert isinstance(scaleArrowColor, tuple), "scaleArrowColor: Please use a tuple like (min, max)!"
            assert len(scaleArrowColor)==2, "scaleArrowColor: Please use a tuple like (min, max)!"
            print(f"Temperature gradient - color_scale: {scaleArrowColor[0]} - {scaleArrowColor[1]}")
            plt.clim(*scaleArrowColor)
        plt.colorbar(image, cmap=plt.cm.jet, label="Temperature [K]", spacing=barSpacing, format="%1.2e")#, ticks=cbticks)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_title('Temperature Gradient')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim((0-self.L*overdrawFactor, self.L*(1+overdrawFactor)))
        ax.set_ylim((0-self.L*overdrawFactor, self.L*(1+overdrawFactor)))
        if saveFig:
            saveName = os.path.join(self._plotDir, name + '_TemperatureGradient.png')
            plt.savefig(saveName)
        
    def plotFlux(self, name, saveFig=True, barSpacing="uniform", scaleArrowColor:tuple=None):
        fig,ax=plt.subplots(1,1)
        self.drawMesh(False)
        plt.tricontourf(self.coords[0],self.coords[1],np.squeeze(self.T),20,cmap='jet',alpha=0.2)
        x, y = np.zeros(self.numberElementsTotal), np.zeros(self.numberElementsTotal)
        u, v = np.zeros_like(x), np.zeros_like(y)
        for i, e in enumerate(self.elements):
            x[i], y[i] = e.center
            u[i], v[i] = e.flux
        M = np.sqrt(u*u + v*v)
        image = plt.quiver(x,y,u,v,M,cmap=plt.cm.jet)
        if not scaleArrowColor:
            # Determine max/min arrow lengths
            minArrow, maxArrow = min(M), max(M)
            if maxArrow - minArrow < 1e-4*maxArrow:
                minArrow = int(minArrow*1e-3)*1e3
            print(f"Flux color_scale: {minArrow} - {maxArrow}")
            plt.clim(minArrow,maxArrow)
        else:
            assert isinstance(scaleArrowColor, tuple), "scaleArrowColor: Please use a tuple like (min, max)!"
            assert len(scaleArrowColor)==2, "scaleArrowColor: Please use a tuple like (min, max)!"
            print(f"Flux gradient - color_scale: {scaleArrowColor[0]} - {scaleArrowColor[1]}")
            plt.clim(*scaleArrowColor)
        plt.colorbar(image, cmap=plt.cm.jet, label="Flux [W/m²]", spacing=barSpacing, format="%1.2e")#, ticks=cbticks)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_title('Flux')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim((0-self.L*overdrawFactor, self.L*(1+overdrawFactor)))
        ax.set_ylim((0-self.L*overdrawFactor, self.L*(1+overdrawFactor)))
        if saveFig:
            saveName = os.path.join(self._plotDir, name + '_Flux.png')
            plt.savefig(saveName)
            
    def plotTemperatureGradientOld(self, name, saveFig=True, barSpacing="uniform", scale=None):
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
            
        image = plt.quiver(x,y,u,v,M,cmap=plt.cm.jet, scale=scale)
        # Determine max/min arrow lengths
        minArrow, maxArrow = min(M), max(M)
        print(f"color_scale: {minArrow} - {maxArrow}")
        plt.clim(minArrow,maxArrow)
        plt.colorbar(image, cmap=plt.cm.jet, label="Temperature [K]", spacing=barSpacing, format="%3.3f s")#, ticks=cbticks)
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax.set_title('Temperature Gradient')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if saveFig:
            saveName = os.path.join(self._plotDir, name + '_TemperatureGradient.png')
            plt.savefig(saveName)
            