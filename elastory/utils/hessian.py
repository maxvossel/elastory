###########
# Hessian #
###########

import itertools as it

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist, squareform

from elastory.utils.matrix import calc_diff_mat, nb_diag_hessian, nb_hessian, symmetrize


# TODO: check if this is still needed -> it is identical to calc_hessian_with_nzeqd at the initial positions and we only use it there?
def calc_hessian(pos: np.ndarray, laplacian: np.ndarray):
    """
    Calculates the hessian (analytically) of the networks potential engergy
    at the current positions.
    Args:
        pos_now: (N,D) np.ndarray
                 Beads current positions

    Returns:
        hessian: (N,N,D,D) np.ndarray
                 Hessian matrix of the network
        dist_mat_r: (N,N) np.ndarray
                    pairwise distance matrix
        diff_mat: (N,N,D) np.ndarray
                    matrix of differences in pairwise distances

    """

    # pairwise distance
    dist_mat_r = squareform(pdist(pos))

    diff_mat_triu = calc_diff_mat(pos)
    diff_mat = symmetrize(diff_mat_triu)
    hessian_no_diag = nb_hessian(diff_mat, dist_mat_r, laplacian)
    hessian = nb_diag_hessian(hessian_no_diag)

    return hessian, dist_mat_r, diff_mat


##############################################
# Hessian with nonzero equilibrium distances #
##############################################

# xy, xz, yz
combinations = [*it.combinations(range(3), 2)]
combinations = np.array(combinations)


@njit(fastmath=True)
def give_3(arr):
    """numba equivalent of numpy.c_[arr, arr, arr]"""
    return np.stack((arr, arr, arr)).T


@njit(fastmath=True)
def distances_for_hessian(pos, connected):
    # Current distances, evaluated at each position again
    dxyz = pos[connected[1]] - pos[connected[0]]
    dxyz2 = dxyz**2
    dr = np.sqrt(np.sum(dxyz**2, axis=1))
    dr2 = give_3(dr**2)
    dr3 = give_3(dr**3)
    return dxyz, dxyz2, dr, dr2, dr3


@njit(fastmath=True)
def diagonal_elements(eq_dist, dxyz2, dr2, dr3):
    diag = -(1 + eq_dist / dr3 * (dxyz2 - dr2))
    return diag


@njit(fastmath=True)
def offdiagonal_elements(eq_dist, dxyz, dr3, combinations):
    offdiag = np.empty_like(eq_dist.T)
    for i in prange(3):
        offdiag[i] = dxyz[:, combinations[i][0]] * dxyz[:, combinations[i][1]]
    offdiag = offdiag.T
    offdiag /= dr3
    offdiag *= -eq_dist
    return offdiag


@njit(fastmath=True)
def nb_fill_hessian(N, D, combinations, connected, diag, offdiag):
    conn = connected * 3
    hess = np.zeros((N * D, N * D))
    for i in prange(3):
        j = combinations[i][0]
        k = combinations[i][1]
        ms = conn[0]
        ns = conn[1]
        for c in range(len(connected[0])):
            m = ms[c]
            n = ns[c]
            # diagonal elements first
            hess[m + i, n + i] = diag[c, i]
            hess[n + i, m + i] = diag[c, i]
            # offdiagonal elements
            hess[m + j, n + k] = offdiag[c, i]
            hess[m + k, n + j] = offdiag[c, i]
            hess[n + j, m + k] = offdiag[c, i]
            hess[n + k, m + j] = offdiag[c, i]
    return hess


@njit(fastmath=True)
def calc_hessian_with_nzeqd(
    eq_dist: np.ndarray, pos: np.ndarray, connected: np.ndarray
) -> np.ndarray:
    """
    Calculate the Hessian matrix with nonzero equilibrium distances.
    Args:
        eq_dist: (N,) np.ndarray
                 equilibrium distances
        pos: (N,D) np.ndarray
             current positions
        connected: (2,M) np.ndarray
                   connected beads
    """
    eq_dist = give_3(eq_dist)
    N, D = pos.shape
    dxyz, dxyz2, dr, dr2, dr3 = distances_for_hessian(pos, connected)
    diag = diagonal_elements(eq_dist, dxyz2, dr2, dr3)
    offdiag = offdiagonal_elements(eq_dist, dxyz, dr3, combinations)
    hess = nb_fill_hessian(N, D, combinations, connected, diag, offdiag)
    hess = nb_diag_hessian(hess)
    return hess


def full_potential(pos_now, dist_mat0, conn):
    """
    Calculate the potential energy of the network in its current positions.
    Args:
        pos_now     : (N,D) np.ndarray
                      positions of beads
        dist0_mat   : (N,N) np.ndarray
                      original distances (resting lengths of springs)
        conn        : (N,N) np.ndarray
                      matrix describing connected beads of network
    Returns:
        pot_energy  : (float)
                      potential energy of the network in its current state
    """
    # conn = connectivity(lapl)
    dist_mat = squareform(pdist(pos_now))
    diff = dist_mat - dist_mat0  # only differences in pairwise distances
    diff *= conn  # only connected pairs
    diff = np.triu(diff)  # take connected pairs only once
    pot_energy = (diff**2).sum()  # sum all contributions
    return pot_energy
