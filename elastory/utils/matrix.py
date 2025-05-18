######################
# basic matrix tools #
######################

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist, squareform


@njit
def symmetrize(mat):
    """
    Symmetrize a matrix.
    Args:
        (ndarray): mat, matrix
    Returns:
        (ndarray): result, symmetrized matrix
    """
    N = mat.shape[0]
    result = np.zeros_like(mat)
    for n in range(N):
        result[n] = mat[n] + np.transpose(mat[n])
    return result


@njit
def anti_symmetrize(mat):
    """
    Anti-symmetrize a matrix.
    Args:
         (ndarray): mat, matrix
    Returns:
         (ndarray): result, anti-symmetrized matrix
    """
    N = mat.shape[0]
    result = np.zeros_like(mat)
    for n in range(N):
        result[n] = mat[n] - np.transpose(mat[n])
    return result


@njit
def nb_outer(vec):
    """
    Outer product / cross product of a vector with itself.
    Args:
        (ndarray): vec, vector
    Returns:
        (ndarray): result, outer product
    """
    N = vec.shape[0]
    result = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, i + 1):
            x = vec[i] * vec[j]
            result[i, j] = x
            result[j, i] = x
    return result


@njit
def nb_hessian(diff_mat, dist_mat_r, laplacian):
    """
    Calculate the hessian matrix of the network.
    Args:
        (ndarray): diff_mat, difference matrix
        (ndarray): dist_mat_r, pairwise distance matrix
        (ndarray): laplacian, laplacian matrix
    Returns:
        (ndarray): result, hessian matrix
    """
    D, N, _ = diff_mat.shape
    result = np.zeros((D * N, D * N))
    for i in prange(0, N):
        k = D * i
        for j in range(0, i + 1):
            m = D * j
            r_ij = dist_mat_r[i, j]
            if r_ij == 0:
                result[k : k + D, m : m + D] = 0
            else:
                r_ij2 = r_ij**2
                c = laplacian[i, j] / r_ij2
                result[k : k + D, m : m + D] = nb_outer(diff_mat[:, i, j]) * c
    return result + result.T


@njit
def nb_diag_hessian(mat):
    """
    Fill the diagonal elements of the hessian matrix with the sum of the
    offdiagonal elements along each row.
    Args:
        (ndarray): mat, hessian matrix
    """
    result = mat.copy()

    N = result.shape[0]
    for i in range(N):
        for j in range(3):
            k = i - i % 3 + j
            result[i, k] = -result[i][j::3].sum()

    return result


@njit
def calc_diff_mat(pos: np.ndarray):
    """
    Returns the difference matrix of the positions
    Args:
        (ndarray): pos, bead positions
    Returns:
        (ndarray): result, difference matrix

    """
    N, D = pos.shape
    result = np.zeros((D, N, N))
    for i in range(0, N):
        for j in range(0, i + 1):
            result[:, i, j] = pos[j] - pos[i]
    return result


def connected_beads(connectivity: np.ndarray) -> np.ndarray:
    """
    Returns the indices of the connected beads
    Args:
        (ndarray): connectivity, connectivity matrix
    Returns:
        (ndarray): connected, indices of connected beads
    """
    return np.array(np.where(np.triu(connectivity) == 1))


def clear_diagonal(mat: np.ndarray):
    """
    Set all values on the diagonal of a matrix to zero
    Args:
        (ndarray): mat, matrix
    Returns:
        (ndarray): mat, matrix with diagonal values set to zero
    """
    result = np.copy(mat)
    np.fill_diagonal(result, 0)
    return result


def populate_diagonal(mat: np.ndarray):
    """
    Populates the diagonal with the correct number of bonds
    Args:
        (ndarray): mat, Kirchhoff matrix
    Returns:
        (ndarray): mat, Kirchhoff matrix
    """
    result = mat.copy()
    diag = result.diagonal()
    diag.setflags(write=True)
    diag[:] = -result.sum(axis=0)
    return result


def connectivity(laplacian: np.ndarray) -> np.ndarray:
    """
    Returns the connectivity matrix from the laplacian matrix.
    Args:
        (ndarray (N,N) ): laplacian matrix
    Returns:
        (ndarray (N,N) ): connectivity matrix
    """
    return np.array((laplacian == -1), dtype=int)


def laplacian(pos: np.ndarray, cutoff_length: float) -> np.ndarray:
    """
    Wherever the pairwise distance between two beads is smaller than the
    cutoff length, we consider those beads as connected. This is
    represented in the matrix with positive (+1) entries.
    The diagonal represents the number of connections of each bead.
    The sum over the rows is 0, so the diagonal contains d times -1 where d
    is the number of beads connected to the bead.
    Args:
        (ndarray (N,D) ): bead positions
        (float)         : cutoff length
    Returns:
        (ndarray (N,N) ): laplacian matrix
    """
    dist_mat_r = squareform(pdist(pos))
    laplacian = np.array(dist_mat_r < cutoff_length, dtype=np.int64)
    laplacian = clear_diagonal(laplacian) * (-1)
    laplacian = populate_diagonal(laplacian)

    return laplacian
