import numpy as np
import matplotlib.pylab as plt
import ot

######################## PROGRAMME ISSU LIBRAIRIE PYTHON #######################

def transport_pyt_lib():

    #-------------------------- DEFINITION DES PARAMETRES --------------------------
    N = 100
    x = np.arange(N, dtype=np.float64)
    print("------------------------ Paramètre x ----------------------------------")
    print(x)
    print("-----------------------------------------------------------------------")

    #-------------------------- GAUSSIENNE DISTRIBUTION ----------------------------
    G1 = ot.datasets.make_1D_gauss(N, m=20, s=5)
    print("------------------------ Gaussienne 1 ---------------------------------")
    print(G1)
    print("-----------------------------------------------------------------------")
    G2 = ot.datasets.make_1D_gauss(N, m=60, s=8)
    print("------------------------ Gaussienne 2 ---------------------------------")
    print(G2)
    print("-----------------------------------------------------------------------")

    #-------------------------- MATRICE CONTENANT LES DISTRIBUTIONS ----------------
    G = np.vstack((G1, G2)).T
    n_distributions = G.shape[1]
    print("------------------------ Matrice des gaussiennes ----------------------")
    print(G)
    print("-----------------------------------------------------------------------")

    #------------------------- AFFICHAGE DES 2 DISTRIBUTIONS -----------------------

    plt.figure(figsize=(7,7))
    for i in range(n_distributions):
        plt.plot(x, G[:, i])
    plt.title('Distributions')
    plt.tight_layout()
    plt.show()

    #------------------------ MATRICE DES COUTS ET NORMALISATION ------------------
    M = ot.utils.dist0(N)
    M /= M.max()

    #----------------------- WASSERSTEIN BARYCENTERS ------------------------------

    alpha = 0.2
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = G.dot(weights)

    # wasserstein
    reg = 1e-3
    bary_wass = ot.bregman.barycenter(G, M, reg, weights)

    plt.figure(figsize=(7,7))
    plt.plot(x, bary_l2, 'r', label='l2')
    plt.plot(x, bary_wass, 'g', label='Wasserstein')
    plt.legend()
    plt.title('Barycenters')
    plt.tight_layout()

######################## PROGRAMME WASSERSTEIN HEAT KERNEL #########################

def wasserstein_barycenters(alpha,gamma):

    #-------------------------- DEFINITION DES PARAMETRES --------------------------
    N = 100
    x = np.arange(N, dtype=np.float64)
    print("------------------------ Paramètre x ----------------------------------")
    print(x)
    print("-----------------------------------------------------------------------")

    #-------------------------- GAUSSIENNE DISTRIBUTION ----------------------------
    G1 = ot.datasets.make_1D_gauss(N, m=20, s=5)
    print("------------------------ Gaussienne 1 ---------------------------------")
    print(G1)
    print("-----------------------------------------------------------------------")
    G2 = ot.datasets.make_1D_gauss(N, m=60, s=8)
    print("------------------------ Gaussienne 2 ---------------------------------")
    print(G2)
    print("-----------------------------------------------------------------------")

    #-------------------------- MATRICE CONTENANT LES DISTRIBUTIONS ----------------
    G = np.vstack((G1, G2)).T
    n_distributions = G.shape[1]
    print("------------------------ Matrice des gaussiennes ----------------------")
    print(G)
    print("-----------------------------------------------------------------------")

    #------------------------- AFFICHAGE DES 2 DISTRIBUTIONS -----------------------

    plt.figure(figsize=(7,7))
    for i in range(n_distributions):
        plt.plot(x, G[:, i])
    plt.title('Distributions')
    plt.tight_layout()
    plt.show()

    #----------------------- WASSERSTEIN BARYCENTERS HEAT KERNEL -------------------

    if (alpha <= 1 and alpha >= 0):

        weights = np.array([1 - alpha, alpha])

        V = np.ones((N,2))
        W = np.ones((N,2))
        D = np.zeros((N,2))

        a = 1./N

        #On commence l'itération sur le C_i
        for j in range(2):

            MU = np.ones(N)

            for i in range(2):
                HT = heat_kernel_1D(a*V[:,i],gamma)
                W[:,i] = np.divide(G[:,i],HT)# ,where = HT != 0)
                D[:,i] = V[:,i] * heat_kernel_1D(a*W[:,i],gamma)
                MU_NEXT = MU*np.power(D[:,i],weights[i])

            for i in range(2):
                V[:,i] = np.divide(V[:,i]*MU,D[:,i])#, where = D[:,i] != 0)

            MU = MU_NEXT
    else :
        print("ALPHA MUST BE IN [O,1], RESTART, NO PAIN NO GAME")


    plt.figure(figsize=(7,7))
    plt.plot(x, MU_NEXT, 'b', label='Wasserstein')
    plt.legend()
    plt.title('Barycenters')
    plt.tight_layout()

def heat_kernel_1D(x,t):
    HT = np.zeros(np.size(x))
    for i in range(np.size(x)):
        HT[i] = np.exp((-x[i]**2)/(2*t))
    return HT
