import numpy as np
import matplotlib.pylab as plt
from math import*
import ot
import ot.plot




def main_plan_trans(gamma):

    N = 100
    x = np.linspace(0,1,N)
    G = matrix_gaussienne(N, x)
    

    D_V, D_W = sinkhorn(G, gamma)
    H = heat_kernel(N, gamma)
    pi = np.dot(D_V, np.dot(H, D_W))  
    plt.figure(1, figsize=(5, 5))
    plt.plot(x, G[:,0], 'b', label='Source distribution')
    plt.plot(x, G[:,1], 'r', label='Target distribution')
    plt.legend()
    ot.plot.plot1D_mat(G[:,0], G[:,1], pi, 'Matrice du plan de transport optimal')
    plt.show()

    return 0



def sinkhorn(G, gamma):

    N = np.shape(G)[0]
    k = np.shape(G)[1]

    a = 1./N

    v = np.ones(N)
    w = np.ones(N)


    for i in range(20):
        v = np.divide(G[:,0], H(a*w, N, gamma))
        w = np.divide(G[:,1], H(a*v, N, gamma))


    D_V = np.diag(v)
    D_W = np.diag(w)
    return D_V, D_W



def heat_kernel(N, gamma):

    t = np.linspace(0,1,N)
    X,Y = np.meshgrid(t,t)
    dist = np.abs(X - Y)**2
    H = np.exp(-(dist**2)/(2*gamma))

    return H

def H(x, N, gamma):
    return np.dot(heat_kernel(N,gamma),x)

def matrix_gaussienne(N, x):
    
    G1 = np.zeros(N)
    G2 = np.zeros(N)

    for i in range(N):
        
        G1[i] = np.exp(-(1./2)*((x[i] - 0.2)/0.08)**2) 
        G2[i] = np.exp(-(1./2)*((x[i] - 0.8)/0.08)**2)
        
    G1 = G1/np.linalg.norm(G1)
    G2 = G2/np.linalg.norm(G2)


    G = np.vstack((G1,G2)).T

    return G