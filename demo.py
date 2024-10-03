import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def top(nelx, nely, volfrac, penal, rmin):
    # Initialize design variables
    x = volfrac * np.ones((nely, nelx))
    change = 1
    loop = 0

    KE = lk()

    # Prepare filter
    H, Hs = prepare_filter(nelx, nely, rmin)

    while change > 0.01:
        loop += 1
        # Finite element analysis
        sK, iK, jK = build_global_stiffness(nelx, nely, x, penal, KE)
        K = coo_matrix((sK, (iK, jK))).tocsc()

        U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))
        F = np.zeros((2 * (nely + 1) * (nelx + 1), 1))
        F[1, 0] = -1  # Loading condition

        # Solve for displacements
        freedofs = np.setdiff1d(np.arange(2 * (nely + 1) * (nelx + 1)), np.arange(2 * (nely + 1)))
        U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])

        # Objective function and sensitivity analysis
        c = 0
        dc = np.zeros((nely, nelx))
        dv = np.ones((nely, nelx))

        for elx in range(nelx):
            for ely in range(nely):
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                Ue = np.concatenate([U[2*n1:2*n1+2], U[2*n2:2*n2+2], U[2*n2+2:2*n2+4], U[2*n1+2:2*n1+4]])
                c += (x[ely, elx] ** penal) * np.dot(Ue.T, np.dot(KE, Ue))
                dc[ely, elx] = -penal * x[ely, elx] ** (penal - 1) * np.dot(Ue.T, np.dot(KE, Ue))

        # Filtering/modification of sensitivities
        dc = filter_sensitivities(nelx, nely, dc, H, Hs, x)

        # Design update by optimality criteria
        xnew = update_design(x, dc, dv, volfrac)

        change = np.max(np.abs(xnew - x))
        x = xnew

        # Print results
        print(f'Iteration: {loop}, Objective: {c:.4f}, Volume: {np.mean(x):.3f}, Change: {change:.3f}')

        # Plot densities
        plt.imshow(1 - x, cmap='gray', origin='lower')
        plt.colorbar()
        plt.show()

def lk():
    E = 1
    nu = 0.3
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

def prepare_filter(nelx, nely, rmin):
    H = np.zeros((nelx * nely, nelx * nely))
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(i - int(np.ceil(rmin)), 0), min(i + int(np.ceil(rmin)), nelx)):
                for l in range(max(j - int(np.ceil(rmin)), 0), min(j + int(np.ceil(rmin)), nely)):
                    col = k * nely + l
                    H[row, col] = max(0, rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2))
    Hs = np.sum(H, axis=1)
    return H, Hs

def build_global_stiffness(nelx, nely, x, penal, KE):
    # n_dofs = 2 * (nelx + 1) * (nely + 1)
    iK = np.zeros((8 * nelx * nely), dtype=int)
    jK = np.zeros((8 * nelx * nely), dtype=int)
    sK = np.zeros((8 * nelx * nely))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
            iK[8*(ely*nelx+elx):8*(ely*nelx+elx+1)] = edof
            jK[8*(ely*nelx+elx):8*(ely*nelx+elx+1)] = edof
            sK[8*(ely*nelx+elx):8*(ely*nelx+elx+1)] = (x[ely, elx] ** penal) * KE.flatten()

    return sK, iK, jK


def filter_sensitivities(nelx, nely, dc, H, Hs, x):
    dc_filtered = np.zeros_like(dc)
    for i in range(nelx):
        for j in range(nely):
            dc_filtered[j, i] = np.sum(H[i * nely + j, :] * dc.flatten()) / Hs[i * nely + j]
    return dc_filtered

def update_design(x, dc, dv, volfrac):
    l1 = 0
    l2 = 1e9
    move = 0.2
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if np.sum(xnew) - volfrac * np.prod(xnew.shape) > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

# Example usage:
top(60, 30, 0.5, 3, 1.5)
