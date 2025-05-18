import numpy as np
from scipy.spatial.distance import pdist, squareform


def degrees_of_freedom(D: int) -> int:
    """
    The number of non deforming motions in D dimensions, see:
    https://www.wikiwand.com/en/Degrees_of_freedom_(mechanics)
    #/Motions_and_dimensions
    Args:
        D: The spatial dimension.
    Returns:
        The number of non deforming motions.
    """
    return D + int(D * (D - 1) / 2)


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
    dist_mat = squareform(pdist(pos_now))
    diff = dist_mat - dist_mat0  # only differences in pairwise distances
    diff *= conn  # only connected pairs
    diff = np.triu(diff)  # take connected pairs only once
    pot_energy = (diff**2).sum()  # sum all contributions
    return pot_energy
