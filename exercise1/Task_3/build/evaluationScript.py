import numpy as np
import matplotlib.pyplot as plt

it = 8

## Double section ## 

h = open('Compare_double.txt', 'r')
content = h.readlines()

val64_d = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val128_d = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val256_d = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val512_d = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
iterations = [np.power(10,i) for i in range(it)] 

counter = 0
for line in content:
    a = []
    a.extend([np.float64(item) for item in line.split(   )])

    if a[0] == 64:
        for j in range(4):
           val64_d[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 128:
        for j in range(4):
           val128_d[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 256:
        for j in range(4):
           val256_d[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 512:
        for j in range(4):
           val512_d[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
            
## Float section ##
            
h = open('Compare_float.txt', 'r')
content = h.readlines()

val64_f = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val128_f = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val256_f = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])
val512_f = np.array([[0.0,0.0,0.0,0.0] for x in range(it)])

counter = 0
for line in content:
    a = []
    a.extend([np.float64(item) for item in line.split(   )])

    if a[0] == 64:
        for j in range(4):
           val64_f[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 128:
        for j in range(4):
           val128_f[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 256:
        for j in range(4):
           val256_f[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
    if a[0] == 512:
        for j in range(4):
           val512_f[counter,j] = a[j+2]
        counter += 1
        if counter == it:
            counter = 0
            
          
plt.figure()
plt.loglog(iterations, val64_f[:,0],'--ob',label = 'norm2_residual float')
plt.loglog(iterations, val64_f[:,1],'--og',label = 'normMax_residual float')
plt.loglog(iterations, val64_f[:,2],'--or',label = 'norm2_error float')
plt.loglog(iterations, val64_f[:,3],'--oc',label = 'normMax_error float')
plt.loglog(iterations, val64_d[:,0],'-ob',label = 'norm2_residual double')
plt.loglog(iterations, val64_d[:,1],'-og',label = 'normMax_residual double')
plt.loglog(iterations, val64_d[:,2],'-or',label = 'norm2_error double')
plt.loglog(iterations, val64_d[:,3],'-oc',label = 'normMax_error double')
plt.title('Comparison double - float, resolution 64', size = 25)
plt.xlabel('number of iterations [-]', size=20)
plt.ylabel('error / residual [-]', size = 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.grid()
plt.legend(prop={'size': 15})
plt.savefig('compare_64.png')

plt.figure()
plt.loglog(iterations, val128_f[:,0],'--ob',label = 'norm2_residual float')
plt.loglog(iterations, val128_f[:,1],'--og',label = 'normMax_residual float')
plt.loglog(iterations, val128_f[:,2],'--or',label = 'norm2_error float')
plt.loglog(iterations, val128_f[:,3],'--oc',label = 'normMax_error float')
plt.loglog(iterations, val128_d[:,0],'-ob',label = 'norm2_residual double')
plt.loglog(iterations, val128_d[:,1],'-og',label = 'normMax_residual double')
plt.loglog(iterations, val128_d[:,2],'-or',label = 'norm2_error double')
plt.loglog(iterations, val128_d[:,3],'-oc',label = 'normMax_error double')
plt.title('Comparison double - float, resolution 128', size = 25)
plt.xlabel('number of iterations [-]', size=20)
plt.ylabel('error / residual [-]', size = 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.grid()
plt.legend(prop={'size': 15})
plt.savefig('compare_128.png')

plt.figure()
plt.loglog(iterations, val256_f[:,0],'--ob',label = 'norm2_residual float')
plt.loglog(iterations, val256_f[:,1],'--og',label = 'normMax_residual float')
plt.loglog(iterations, val256_f[:,2],'--or',label = 'norm2_error float')
plt.loglog(iterations, val256_f[:,3],'--oc',label = 'normMax_error float')
plt.loglog(iterations, val256_d[:,0],'-ob',label = 'norm2_residual double')
plt.loglog(iterations, val256_d[:,1],'-og',label = 'normMax_residual double')
plt.loglog(iterations, val256_d[:,2],'-or',label = 'norm2_error double')
plt.loglog(iterations, val256_d[:,3],'-oc',label = 'normMax_error double')
plt.title('Comparison double - float, resolution 256', size = 25)
plt.xlabel('number of iterations [-]', size=20)
plt.ylabel('error / residual [-]', size = 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.grid()
plt.legend(prop={'size': 15})
plt.savefig('compare_256.png')

plt.figure()
plt.loglog(iterations, val512_f[:,0],'--ob',label = 'norm2_residual float')
plt.loglog(iterations, val512_f[:,1],'--og',label = 'normMax_residual float')
plt.loglog(iterations, val512_f[:,2],'--or',label = 'norm2_error float')
plt.loglog(iterations, val512_f[:,3],'--oc',label = 'normMax_error float')
plt.loglog(iterations, val512_d[:,0],'-ob',label = 'norm2_residual double')
plt.loglog(iterations, val512_d[:,1],'-og',label = 'normMax_residual double')
plt.loglog(iterations, val512_d[:,2],'-or',label = 'norm2_error double')
plt.loglog(iterations, val512_d[:,3],'-oc',label = 'normMax_error double')
plt.title('Comparison double - float, resolution 512', size = 25)
plt.xlabel('number of iterations [-]', size=20)
plt.ylabel('error / residual [-]', size = 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.grid()
plt.legend(prop={'size': 15})
plt.savefig('compare_512.png')






    
        
        