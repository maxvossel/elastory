import numpy as np
from scipy.linalg import solve

from elastory.exceptions import ConvergenceError
from elastory.response.matrix import cut_blocks, order_blocks, remove_nullspace


def preload_positions(pos_now, beads, preloads, noise_strength=0.0):
    """
    Returns the positions of the selected beads as they are
    subsequently pulled towards their shared center of geometry,
    with optional noise added to the shifts.

    Args:
        pos_now    : (N,D) np.ndarray, current configuration
        beads      : tuple, beads to pull
        preloads   : (P) float, np.ndarray,
                     factor, by how much the distance is changed
                     compared to the original distance
        noise_strength : float, strength of the noise to add (default: 0.0)

    Returns:
        pos_preload : (P,len(beads),D) np.ndarray,
                      positions of selected beads pulled together
    """
    beads = tuple(beads)
    pos = pos_now.copy()
    pos_ind = pos[[*beads]]
    com = np.mean(pos_ind, axis=0)

    dist_vec = com - pos_ind
    shifts = np.array([dist_vec * preload for preload in preloads])

    # Add noise to shifts
    if noise_strength > 0:
        noise = np.random.normal(0, noise_strength, shifts.shape)
        shifts += noise

    return pos_ind + shifts


def simple_qopt(pos_now, pos_preload, hessian, inds, suppress_NullSpaceError=False):
    """
    Simple quadratic optimization without constraints. The part of the
    quadratic form that corresponds to the beads which are kept fixed,
    is taken out and the linear system is only solved for the remainder.
    Args:
        pos_now     : (N,D) float ndarray
                      current positions (x,y,z) of particles
        pos_preload : (N,D) float ndarray
                      positions of particles including those to fix
        hessian     : (N*D, N*D) float ndarray
                      force matrix at old equilibrium positions
        inds        : tuple, int
                      indices of fixed beads
    Returns:
        pos_new     : (N,D) float ndarray
                      new equilibrium positions
    """
    H = hessian.copy()
    inds = inds.copy()
    # inds = tuple(inds)

    c = len(inds)
    N, D = pos_now.shape

    r0 = pos_now.copy()
    r = pos_preload.copy()
    r0 = r0.reshape((-1,))
    r = r.reshape((-1,))

    r0 = order_blocks(r0, inds)
    r = order_blocks(r, inds)
    H = order_blocks(H, inds)

    rc, _ = cut_blocks(r, c)
    r0c, r0s = cut_blocks(r0, c)
    Hc, Hb, Hs = cut_blocks(H, c, symmetric=True)

    a = r0c - rc
    v = Hb @ a + Hs @ r0s
    M = Hs

    if len(inds) < 3:
        M = remove_nullspace(M, suppress_NullSpaceError=suppress_NullSpaceError)

    rs = solve(M, v, assume_a="gen", check_finite=False)
    r = np.r_[rc, rs]
    r = order_blocks(r, inds)

    pos_new = r.reshape((N, D))
    if np.allclose(np.dot(M, rs), v):
        return pos_new
    else:
        raise ConvergenceError("No solution within tolerance found")
