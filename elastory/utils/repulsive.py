#################################
# WCA / repulsive Lennard Jones #
#################################

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

from elastory.utils.matrix import calc_diff_mat

xyz_dict = {"xx": 0, "yy": 1, "zz": 2, "xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}


def dense_square_map(
    N: int,
) -> np.vectorize:
    """
    Returns a vectorized mapping function that maps a 1D array of indices to a 2D array of indices.
    """
    num_pairs = int(N / 2 * (N - 1))
    d2 = [(i, j) for i in range(N) for j in range(i + 1, N)]
    d1 = list(range(num_pairs))
    d1d2 = dict(zip(d1, d2))
    mapping = np.vectorize(d1d2.get)
    return mapping


def square_dense_map(
    N: int,
) -> np.vectorize:
    """
    Returns a vectorized mapping function that maps a 2D array of indices to a 1D array of indices.
    """
    num_pairs = int(N / 2 * (N - 1))
    d2 = [(i, j) for i in range(N) for j in range(i + 1, N)]
    d1 = list(range(num_pairs))
    d2d1 = dict(zip(d2, d1))
    mapping = np.vectorize(d2d1.get)
    return mapping


def find_touching_bead_inds(pos, repulsive_cutoff, mapping, which="square"):
    """
    'which' can be 'square' or 'dense' for 2D or 1D array of indices
    """
    pdp = pdist(pos)
    repulsive_cutoff *= 2
    conn_d1 = np.nonzero(pdp < repulsive_cutoff)[0]
    if conn_d1.shape[0] == 0:
        return None
    else:
        if which == "square":
            conn_d2 = mapping(conn_d1)
            conn = [*zip(*conn_d2)]
        elif which == "dense":
            conn = conn_d1
        return conn


def calc_shifts(pos):
    pdr = squareform(pdist(pos))
    pdr[np.diag_indices_from(pdr)] = 1
    pdx, pdy, pdz = calc_diff_mat(pos)
    return pdr, pdx, pdy, pdz


@njit
def norm_shifts(pdr, pdx, pdy, pdz):
    pdx = pdx + pdx.T
    pdy = pdy + pdy.T
    pdz = pdz + pdz.T
    pex = pdx / pdr
    pey = pdy / pdr
    pez = pdz / pdr
    return pex, pey, pez


@njit
def f_1(a, b, b2):
    return a * (7 * b - 13 * b2)


@njit
def f_2(a, b, b2):
    return a * (b - b2)


@njit
def calc_pure(ea2, f1, f2):
    return ea2 * f2 + (ea2 - 1) * f1


@njit
def calc_mixed(eab, f1, f2):
    return eab * f2 + (eab * f1)


@njit
def fill_pure(H, val, i, j, m):
    H[i + m, j + m] = val
    H[j + m, i + m] = val
    H[i + m, i + m] -= val
    H[j + m, j + m] -= val


@njit
def fill_mixed(H, val, i, j, m, n):
    H[i + m, j + n] = val
    H[i + n, j + m] = val
    H[j + m, i + n] = val
    H[j + n, i + m] = val
    H[i + m, i + n] -= val
    H[i + n, i + m] -= val
    H[j + m, j + n] -= val
    H[j + n, j + m] -= val


@njit
def lj_terms(dr, epsilon, sigma):
    r2 = dr**2

    a = 4 * epsilon / r2
    b = 6 * sigma**6 / r2**3
    b2 = 12 * b**2
    return a, b, b2


def repulsive_hessian(pos, repulsive_cutoff, mapping, epsilon, sigma):
    conn = find_touching_bead_inds(pos, repulsive_cutoff, mapping)
    N, D = pos.shape
    H = np.zeros((N * D, N * D))

    if conn is None:
        return H

    pdr, pdx, pdy, pdz = calc_shifts(pos)
    pex, pey, pez = norm_shifts(pdr, pdx, pdy, pdz)

    for i, j in conn:
        dr = pdr[i, j]
        ex = pex[i, j]
        ey = pey[i, j]
        ez = pez[i, j]

        a, b, b2 = lj_terms(dr, epsilon, sigma)

        f1 = f_1(a, b, b2)
        f2 = f_2(a, b, b2)

        i *= D
        j *= D

        for xyz, ea in zip(("xx", "yy", "zz"), (ex, ey, ez)):
            m = xyz_dict[xyz]
            ea2 = ea**2
            val = calc_pure(ea2, f1, f2)
            fill_pure(H, val, i, j, m)

        for xyz, (ea, eb) in zip(("xy", "yz", "xz"), ((ex, ey), (ey, ez), (ex, ez))):
            m, n = xyz_dict[xyz]
            eab = ea * eb
            val = calc_mixed(eab, f1, f2)
            fill_mixed(H, val, i, j, m, n)

    return H


@njit
def lj_potential(dr, epsilon, sigma):
    r2 = dr**2
    s2 = sigma**2
    a = (s2 / r2) ** 3
    pot = 4 * epsilon * (a**2 - a) + epsilon
    return pot


@njit
def lj_add_terms(conn, pdr, epsilon, sigma):
    pot = 0
    for i in conn:
        dr = pdr[i]
        pot += lj_potential(dr, epsilon, sigma)
    return pot


def repulsive_potential(pos, repulsive_cutoff, epsilon, sigma):
    pdr = pdist(pos)
    repulsive_cutoff *= 2
    conn = np.nonzero(pdr < repulsive_cutoff)[0]

    if conn.shape[0] != 0:
        pot = lj_add_terms(conn, pdr, epsilon, sigma)
        return pot
    else:
        return 0
