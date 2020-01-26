import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from math import*


def main(N, gamma, alpha):
    
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)

    GAUSS1 = matrix_gaussienne_2D(x, y, 0.30, 0.30, 0.1)
    GAUSS_1 = GAUSS1.reshape(N*N)
    GAUSS2 = matrix_gaussienne_2D(x, y, 0.7, 0.7, 0.1)
    GAUSS_2 = GAUSS2.reshape(N*N)
    G = np.vstack((GAUSS_1,GAUSS_2)).T
    ALPHA = np.array([1 - alpha,alpha])
    BARY = barycenter_2D(G, ALPHA, gamma).reshape((N,N))
    
    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,BARY, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BARYCENTRE avec alpha = {}" .format(alpha))
    
    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,GAUSS1, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("INITIAL")
    
    plt.show()
    

    return BARY


def barycenter_2D(G, ALPHA, gamma):
    
    N = np.shape(G)[0]
    k = np.shape(G)[1]
    a = 1./N

    V = np.ones((N, k))
    W = np.ones((N, k))
    D = np.zeros((N, k))

    for j in range(k):
        MU = np.ones(N)
        for i in range(k):
            HEAT_V = H(a*V[:,i], N, gamma)
            W[:,i] = np.divide(G[:,i], HEAT_V)#, where = HEAT_V != 0)
            HEAT_W = H(a*W[:,i], N, gamma)
            D[:,i] = V[:,i] * HEAT_W
            MU = MU * (np.power(D[:,i], ALPHA[i]))

        for l in range(k):
            V[:,l] = np.divide(V[:,l]*MU, D[:,l])#, where = D[:,i] != 0)

    return MU


def H(x, N, gamma):
    return np.dot(heat_kernel_2D(dist_2D(coord_2D(N)), gamma), x)


def heat_kernel_2D(DIST, gamma):
    
    H = np.exp(-DIST/gamma)
    
    return H


def dist_2D(COORD):
    
    N = np.shape(COORD)[0]
    DIST = np.zeros((N*N, N*N))
    
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for m in range(N):
                    k1 = i + j*N
                    k2 = l + m*N
                    DIST[k1, k2] = np.sqrt((COORD[i, j, 0] - COORD[l, m, 0])**2 + (COORD[i, j, 1] - COORD[l, m, 1])**2)

    return DIST

def coord_2D(n):
    
    N = int(np.sqrt(n))
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    COORD = np.zeros((N, N, 2))
    
    for i in range(N):
        for j in range(N):
            COORD[i, j, 0] = x[i]
            COORD[i, j, 1] = y[j]
            
    return COORD


def matrix_gaussienne_2D(x, y, xm, ym, s):
    
    N = np.size(x)
    G = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            G[i][j] = np.exp((-1./2)*(((x[i]-xm)/s)**2 + ((y[j]-ym)/s)**2))
            
    G = G/np.linalg.norm(G)


    return G