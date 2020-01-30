import numpy as np
from math import*
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

######################### PROGRAMME POUR GAUSSIENNE 1D #########################

def main_1D(N,alpha,gamma):

    x = np.linspace(0,1,N)
    G1 = gaussienne_1D(N,x,0.25,0.01)
    G2 = gaussienne_1D(N,x,0.75,0.01)

    G1 = G1/np.linalg.norm(G1)
    G2 = G2/np.linalg.norm(G2)

    MU = wasserstein_barycenters_1D(G1,G2,alpha,gamma)

    affichage_1D_1(G1,G2)
    affichage_1D_2(MU)

    return G1,G2, MU

def gaussienne_1D(N,x,x0,sig1):

    return np.exp((-(x-x0)**2)/(2*sig1))

def wasserstein_barycenters_1D(S1,S2,alpha,gamma):

    if (alpha <= 1 and alpha>=0):

        N = np.size(S1)

        #Initialisation du tableau des S, alpha, v w et a

        S = np.zeros((N,2))
        S[:,0] = S1
        S[:,1] = S2

        ALPHA = np.zeros(2)
        ALPHA[0] = 1-alpha
        ALPHA[1] = alpha

        V = np.ones((N,2))
        W = np.ones((N,2))
        D = np.zeros((N,2))

        a = 1./N

        #On commence l'itération sur le C_i
        for j in range(2):

            MU = np.ones(N)

            for i in range(2):
                HT = heat_kernel_1D(a*V[:,i],gamma)
                W[:,i] = np.divide(S[:,i],HT,where = HT != 0)
                D[:,i] = V[:,i] * heat_kernel_1D(a*W[:,i],gamma)
                MU = MU*np.power(D[:,i],ALPHA[i])

            for i in range(2):
                V[:,i] = np.divide(V[:,i]*MU,D[:,i], where = D[:,i] != 0)

        return MU

    else :
        print("ALPHA MUST BE IN [O,1], RESTART, NO PAIN NO GAME")

def affichage_1D_1(G1,G2):

    x = np.linspace(0,1,np.shape(G1)[0])

    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)#, projection = '3d')
    #plt.ylim(0,1)
    ax.plot(x,G1)
    ax.plot(x,G2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Transport Optimal")

    plt.show()

def affichage_1D_2(G1):

    x = np.linspace(0,1,np.shape(G1)[0])

    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)#, projection = '3d')
    ax.plot(x,G1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Transport Optimal")

    plt.show()

######################### PROGRAMME POUR GAUSSIENNE 2D #########################
def main_2D(N,sd):

    G1 = gaussienne_2D(N,0.25,0.25,0.01)
    G2 = gaussienne_2D(N,0.75,0.75,0.01)

    for t in np.arange(0,1.1,0.1):
        t = round(t,3)
        MU = wasserstein_barycenters_2D(G1,G2,t,sd)
        #np.savetxt("MU_alpha_=_{}.txt".format(t),GG1)
        affichage_2D(MU)

    return MU

def gaussienne_2D(N,x0,y0,sig1):

    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)

    G1 = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            G1[i,j] = np.exp(-(((x[i]-x0)**2)/(2*sig1) + ((y[j]-y0)**2)/(2*sig1)))

    G1 = G1/np.linalg.norm(G1)

    return G1

def distance(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def heat_kernel_2D(x,y,t):
    return (1./4*pi*t) * np.exp(-distance(x,y)**2/(4*t))

def heat_kernel_1D(x,t):
    return (1./np.sqrt(4*pi*t)) * np.exp(-x**2/(4*t))

def wasserstein_barycenters_2D(SS1,SS2,alpha,sd):

    if (alpha <= 1 and alpha>=0):

        N = np.shape(SS1)[0]*np.shape(SS1)[1]
        t = sd/2
        #Transformer skeleton en densité de probas
        S1 = SS1.reshape(N)
        S2 = SS2.reshape(N)
        S1 = S1/np.linalg.norm(S1)
        S2 = S2/np.linalg.norm(S2)

        #Initialisation du tableau des S, alpha, v w et a

        S = np.zeros((N,2))
        S[:,0] = S1
        S[:,1] = S2

        ALPHA = np.zeros(2)
        ALPHA[0] = 1 - alpha
        ALPHA[1] = alpha

        V = np.ones((N,2))
        W = np.ones((N,2))
        D = np.zeros((N,2))

        a = 1./N

        #On commence l'itération sur le C_i
        for j in range(2):

            MU = np.ones(N)

            for i in range(2):
                HT = heat_kernel_1D(a*V[:,i],t)
                W[:,i] = np.divide(S[:,i],HT,where = HT != 0)
                D[:,i] = V[:,i] * heat_kernel_1D(a*W[:,i],t)
                MU = MU*np.power(D[:,i],ALPHA[i])

            for i in range(2):
                V[:,i] = np.divide(V[:,i]*MU,D[:,i], where = D[:,i] != 0)

        MU = MU.reshape(np.shape(SS1)[0],np.shape(SS1)[1])

        return MU

    else :
        print("ALPHA MUST BE IN [O,1], RESTART, NO PAIN NO GAME")

def affichage_2D(G1):

    x = np.linspace(0,1,np.shape(G1)[0])
    y = np.linspace(0,1,np.shape(G1)[1])

    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,G1, cmap = 'plasma')
    #ax.contour(X,Y,G2, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Transport Optimal")

    plt.show()
