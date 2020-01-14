import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.collections import PolyCollection
import ot


def python_wasserstein():

    #%% parameters

    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    #%% barycenter computation

    alpha = 0.2  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    bary_wass = ot.bregman.barycenter_sinkhorn(A, M, reg, weights)

    plt.figure(2)
    plt.clf()
    plt.subplot(2, 1, 1)
    for i in range(n_distributions):
        plt.plot(x, A[:, i])
    plt.title('Distributions')

    plt.subplot(2, 1, 2)
    plt.plot(x, bary_l2, 'r', label='l2')
    plt.plot(x, bary_wass, 'g', label='Wasserstein')
    plt.legend()
    plt.title('Barycenters')
    plt.tight_layout()

    #%% barycenter interpolation

    n_alpha = 11
    alpha_list = np.linspace(0, 1, n_alpha)


    B_l2 = np.zeros((n, n_alpha))

    B_wass = np.copy(B_l2)

    for i in range(0, n_alpha):
        alpha = alpha_list[i]
        poids = np.array([1 - alpha, alpha])
        print(poids)
        B_l2[:, i] = A.dot(poids)
        B_wass[:, i] = ot.bregman.barycenter_sinkhorn(A, M, reg, poids)

    #%% plot interpolation

    plt.figure(4)
    cmap = plt.cm.get_cmap('viridis')
    verts = []
    zs = alpha_list
    for i, z in enumerate(zs):
        ys = B_wass[:, i]
        verts.append(list(zip(x, ys)))

    ax = plt.gcf().gca(projection='3d')

    poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.set_xlabel('x')
    ax.set_xlim3d(0, n)
    ax.set_ylabel('$\\alpha$')
    ax.set_ylim3d(0, 1)
    ax.set_zlabel('')
    ax.set_zlim3d(0, B_l2.max() * 1.01)
    plt.title('Barycenter interpolation with Wasserstein')
    plt.tight_layout()

    plt.show()
