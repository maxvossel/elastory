import itertools as it

import numpy as np
from scipy.spatial.distance import pdist, squareform

from elastory.utils.conversion import polar_to_cartesian
from elastory.utils.matrix import laplacian


def distribute_beads(
    N: int = 100,
    D: int = 3,
    r_max: float = 20,
    l_min: float = 4,
    max_attempts: int = 10000,
    make_pockets: bool = True,
    cutoff_length: float = 9,
    p0_center: list | np.ndarray = [16, 0, 0],
    p1_center: list | np.ndarray = [-16, 0, 0],
    p0_radius: float = 4,
    p1_radius: float = 4,
) -> np.ndarray:
    """
    Distribute N beads in D dimensions, while satisfying the constraints
    concerning cosecutive distances and distance to center of mass.
    Args:
        N: int, optional
            Number of beads to distribute
        D: int, optional
            Number of dimensions
        r_max: float, optional
            Maximum distance from center of mass
        l_min: float, optional
            Minimum distance between consecutive beads
        max_attempts: int, optional
            Maximum number of attempts to distribute the beads
        make_pockets: bool, optional
            Whether to create pockets around p0_center and p1_center
        cutoff_length: int, optional
            Cutoff length for pairwise distances
        p0_center: list, optional
            Center of pocket 0
        p1_center: list, optional
            Center of pocket 1
        p0_radius: float, optional
            Radius of pocket 0
        p1_radius: float, optional
            Radius of pocket 1
    Returns:
        pos : (N, D) float ndarray
            Positions of the beads
    """
    pos = np.zeros((N, D))

    i = 1
    attempt = 0

    if isinstance(p0_center, list):
        p0_center = np.array(p0_center)
    if isinstance(p1_center, list):
        p1_center = np.array(p1_center)

    while (i < N) and (attempt < max_attempts):
        attempt += 1
        pos_before = pos[i - 1]

        rand_r, rand_theta, rand_phi = np.random.rand(3)
        rand_r += l_min
        rand_theta *= np.pi
        rand_phi *= 2 * np.pi

        x, y, z = polar_to_cartesian(rand_r, rand_theta, rand_phi)
        pos[i] = np.c_[x, y, z] + pos_before

        pair_dist = pdist(pos)
        pair_dist = pair_dist[pair_dist != 0]
        pair_dist_ok = np.all(pair_dist > l_min)

        if make_pockets:
            p0_dist_ok = np.all(np.linalg.norm(pos - p0_center, axis=1) > p0_radius)
            p1_dist_ok = np.all(np.linalg.norm(pos - p1_center, axis=1) > p1_radius)
            pocket_dist_ok = p0_dist_ok and p1_dist_ok
        else:
            pocket_dist_ok = True

        com_dist_ok = np.all(np.linalg.norm(pos, axis=1) < r_max)

        if pair_dist_ok and com_dist_ok and pocket_dist_ok:
            i += 1

    lap = laplacian(pos, cutoff_length)
    num_neighbours_ok = np.all(np.diag(lap) > 2)

    if pair_dist_ok and com_dist_ok and pocket_dist_ok and num_neighbours_ok:
        print(f"Distributed {N} beads in {attempt} attempts.")
        return pos
    else:
        print(f"Tried to distribute {N} beads, failed after {attempt} attempts.")
        raise RuntimeError(
            "Failed to distribute beads, please retry and/or take a look at the parameters"
        )


def merge_bead_positions(
    pos_init_0: np.ndarray,
    pos_init_1: np.ndarray,
    shift: float = 0.5,
    l_min: float = 4,
) -> np.ndarray:
    """
    Merge two sets of initial positions and remove positions that are too close.

    Args:
        pos_init_0: The initial positions of set 0.
        pos_init_1: The initial positions of set 1.
        shift: The shift to apply to the positions.
        l_min: The minimum distance between two positions. (-> refer to the distribute_beads function)

    Returns:
        The merged positions with positions that are too close removed.
    """
    pos_left = np.max(pos_init_0[:, 0])
    pos_right = np.min(pos_init_1[:, 0])
    pos_left = np.array([pos_left, 0, 0])
    pos_right = np.array([pos_right, 0, 0])

    pos_init_combined = np.r_[
        pos_init_0 - shift - 0.25 * pos_left, pos_init_1 + shift - 0.25 * pos_right
    ]

    pos = pos_init_combined.copy()

    pair_dist = np.triu(squareform(pdist(pos)))
    too_close = np.where((pair_dist < l_min) * (pair_dist > 0))

    pos = np.delete(pos, too_close[0], axis=0)

    print(f"Removed {len(too_close[0])} beads that were too close.")
    print(f"Final number of beads: {len(pos)}")
    return pos


def estimate_pocket(
    beads: list[int], pos: np.ndarray, n_pairs: int = 6
) -> tuple[list[int], list[list[int]]]:
    """
    Select a subset of beads based on their pairwise distances.

    This function starts with some initial beads, finds the pairs with the largest
    distances among them (and possibly other beads in the given positions),
    and then keeps only the beads that are part of these selected pairs.

    Parameters:
    -----------
    beads : list[int]
        Indices of some initial beads to consider.
    pos : np.ndarray
        Array of positions for all beads. Shape should be (n_beads, n_dimensions).
    n_pairs : int, optional
        Number of pairs to select based on largest distances.

    Returns:
    --------
    tuple[list[int], list[list[int]]]
        The first element is the list of beads that are part of the selected pairs.
        The second element is a list of pairs of beads that are part of the selected pairs.
    """

    bead_pair_inds = np.argsort(pdist(pos[beads]))[-n_pairs:]
    bead_pairs = np.array(list(it.combinations(beads, 2)), dtype=int)[bead_pair_inds]

    beads = [bead for bead in beads if bead in bead_pairs.flatten()]

    return list(set(beads)), bead_pairs.tolist()
