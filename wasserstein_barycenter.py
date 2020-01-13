import numpy as np
import matplotlib.pylab as plt
import ot
from math import*

def main(gamma):

    N = 100
    G = matrix_gaussienne(N)

    x = np.linspace(0,1,N)
    t = np.arange(0,1.1,0.1)
    plt.figure(figsize=(10,10))
    plt.plot(x,G[:,0],'+', 'r', label = 'Gaussienne départ')
    plt.plot(x,G[:,1],'+', 'r', label = 'Gausiienne arrivée')
    for i in range(np.size(t)):
        alpha = t[i]
        ALPHA = np.array([1 - alpha,alpha])
        BARY = barycenter(G,ALPHA,gamma)
        plt.plot(x, BARY)
    plt.legend()
    plt.title('Barycentres pour gamma = {}'.format(gamma))
    plt.tight_layout()
    plt.show()

    """plt.figure(figsize=(10,10))
    plt.plot(x,G[:,0], label = 'Gaussienne départ')
    plt.plot(x,G[:,1], label = 'Gausiienne arrivée')
    plt.plot(x,np.dot(G,[0.5,0.5]), 'r', label = 'Moyenne')
    plt.title('Gaussiennes et leur moyenne')
    plt.tight_layout()
    plt.show()"""

    return 0

def barycenter(G,ALPHA,gamma):

    N = np.shape(G)[0]
    k = np.shape(G)[1]

    a = 1./N

    V = np.ones((N,k))
    W = np.ones((N,k))
    D = np.zeros((N,k))

    for j in range(k):
        MU = np.ones(N)
        for i in range(k):
            HEAT_V = H(a*V[:,i],N,gamma)
            W[:,i] = np.divide(G[:,i],HEAT_V)#, where = HEAT_V != 0)
            HEAT_W = H(a*W[:,i], N, gamma)
            D[:,i] = V[:,i] * HEAT_W
            MU = MU * (np.power(D[:,i],ALPHA[i]))

        for l in range(k):
            V[:,l] = np.divide(V[:,l]*MU,D[:,l])#, where = D[:,i] != 0)

    return MU

def heat_kernel(N,gamma):

    t = np.linspace(0,1,N)
    X,Y = np.meshgrid(t,t)
    dist = np.abs(X - Y)**2
    H = np.exp(-dist/gamma)

    return H

def H(x,N,gamma):
    return np.dot(heat_kernel(N,gamma),x)

def matrix_gaussienne(N):

    G1 = ot.datasets.make_1D_gauss(N, m = 20, s=4)
    G1 = G1/np.linalg.norm(G1)
    print(np.linalg.norm(G1))

    G2 = ot.datasets.make_1D_gauss(N, m = 60, s=10)
    G2 = G2/np.linalg.norm(G2)
    print(np.linalg.norm(G2))

    G = np.vstack((G1,G2)).T

    return G
