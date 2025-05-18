from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

from elastory.network.network import Network
from elastory.utils.hessian import calc_hessian  # , calc_hessian_with_nzeqd
from elastory.utils.misc import degrees_of_freedom, full_potential
from elastory.utils.repulsive import repulsive_hessian, repulsive_potential


def rgyr(
    pos_arr: np.ndarray,
    beads: np.ndarray | list | int,
) -> np.ndarray:
    """Calculate the radius of gyration of the given beads for either:
        - a single set of coordinates or
        - a full trajectory obtained during optimization
    Args:
        pos_arr : ([preloads,] N, D), float ndarray
                  input positions
        beads   : int, ndarray

    Returns:
        rgyr    : ([preloads,] N, D) float ndarray
                  calculated radius of gyration
    """
    if isinstance(beads, int):
        beads = np.array([beads])
    elif isinstance(beads, list):
        beads = np.array(beads)

    rgyr: np.ndarray
    if pos_arr.ndim == 3:
        pos_bead = pos_arr[:, [*beads]]
        pos_arr_shape = pos_bead.shape
        com = np.mean(pos_bead, axis=1)
        com = com.repeat(beads.shape[0], axis=0).reshape(pos_arr_shape)
        dist_vec = com - pos_bead
        rgyr = np.sum(np.sum(dist_vec**2, axis=1), axis=1)
        rgyr = np.sqrt(rgyr / beads.shape[0])
        return rgyr

    elif pos_arr.ndim == 2:
        pos_bead = pos_arr[[*beads]]
        com = np.mean(pos_bead, axis=0)
        dist_vec = com - pos_bead
        rgyr = np.sum(np.sum(dist_vec**2))
        rgyr = np.sqrt(rgyr / beads.shape[0])
        return rgyr

    else:
        raise ValueError("Dimension Error. Check you input.")


def calculate_spring_pot(
    trajectory: npt.NDArray[np.float64], laplacian: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the potential energy of the springs connecting the beads for the given trajectory.

    Parameters
    ----------
    trajectory : npt.NDArray[np.float64]
        Array of bead positions for each step in the trajectory.
    laplacian : npt.NDArray[np.float64]
        Laplacian matrix of the network.

    Returns
    -------
    npt.NDArray[np.float64]
        Potential energy for each step in the trajectory.
    """
    dist_mat = squareform(pdist(trajectory[0]))
    return np.array([full_potential(rpos, dist_mat, laplacian) for rpos in trajectory])


def calculate_lj_pot(
    trajectory: npt.NDArray[np.float64],
    repulsive_cutoff: float,
    epsilon: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate the potential energy of the overlapping repulsive beads for the given trajectory.

    Parameters
    ----------
    trajectory : npt.NDArray[np.float64]
        Array of bead positions for each step in the trajectory.
    repulsive_cutoff : float
        Cutoff distance for repulsive interactions.
    epsilon : float
        Depth of the potential well.
    sigma : float
        Distance at which the inter-particle potential is zero.

    Returns
    -------
    npt.NDArray[np.float64]
        Lennard-Jones potential energy for each step in the trajectory.
    """
    return np.array(
        [repulsive_potential(rpos, repulsive_cutoff, epsilon, sigma) for rpos in trajectory]
    )


def add_data(
    trajectory: npt.NDArray[np.float64],
    laplacian: npt.NDArray[np.float64],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Add additional data to the response based on specified options.

    Parameters
    ----------
    trajectory : npt.NDArray[np.float64]
        Array of bead positions for each step in the trajectory.
    laplacian : npt.NDArray[np.float64]
        Laplacian matrix of the network.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    dict[str, Any]
        dictionary containing the response data.
    """
    data_options = {
        "save_eigen": False,
        "save_potential": False,
    }
    data_options.update({k: v for k, v in kwargs.items() if k in data_options})

    response_data = {"trajectory": trajectory}

    if data_options["save_eigen"]:
        response_data["eigvals"], response_data["eigvecs"] = calculate_eigen(
            trajectory, laplacian, **kwargs
        )

    if data_options["save_potential"]:
        response_potential = calculate_spring_pot(trajectory, laplacian)
        if kwargs["repulsive_beads"]:
            response_potential += calculate_lj_pot(
                trajectory,
                kwargs["repulsive_cutoff"],
                kwargs["epsilon"],
                kwargs["sigma"],
            )
        response_data["potential"] = response_potential

    return response_data


def calculate_eigen(
    trajectory: npt.NDArray[np.float64],
    laplacian: npt.NDArray[np.float64],
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate and store the eigenvalues and eigenvectors for the given trajectory.

    Parameters
    ----------
    trajectory : npt.NDArray[np.float64]
        Array of bead positions for each step in the trajectory.
    laplacian : npt.NDArray[np.float64]
        Laplacian matrix of the network.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Eigenvalues and eigenvectors for each step in the trajectory.
    """
    eigvals = []
    eigvecs = []

    for rpos in trajectory:
        hessian = calc_hessian(rpos, laplacian)[0]

        if kwargs["repulsive_beads"]:
            # Add repulsive interactions to the Hessian
            hessian += repulsive_hessian(
                rpos,
                kwargs["repulsive_cutoff"],
                kwargs["mapping"],
                kwargs["epsilon"],
                kwargs["sigma"],
            )
        w, v = eigh(hessian)
        eigvals.append(w)
        eigvecs.append(v)

    return np.array(eigvals), np.array(eigvecs)


def eigvals_and_eigvecs(
    net: Network,
    hessian: np.ndarray,
    normed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = net.ND
    dof = degrees_of_freedom(net.D)
    w, v = eigh(hessian, subset_by_index=([dof, n - 1]))
    if normed:
        w /= w.min()
    return w, v
