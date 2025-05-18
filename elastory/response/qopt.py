import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import TqdmExperimentalWarning

from elastory.response.optimizer import preload_positions, simple_qopt
from elastory.utils.hessian import calc_hessian, calc_hessian_with_nzeqd
from elastory.utils.repulsive import repulsive_hessian

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm  # noqa: E402

PROGRESS_BAR = "progress_bar"


def pbar(iterable: Any, progress_bar: bool) -> Any:
    """
    Wrap an iterable with a progress bar if specified.

    Parameters
    ----------
    iterable : Any
        The iterable to wrap.
    progress_bar : bool
        Whether to show a progress bar.

    Returns
    -------
    Any
        The wrapped iterable, with or without a progress bar.
    """
    return tqdm(iterable) if progress_bar else iterable


def _base_qopt(
    pos_init: npt.NDArray[np.float64],
    pos_ind_preload: npt.NDArray[np.float64],
    beads: list[int],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """
    Base function for QOPT calculations.

    Parameters
    ----------
    pos_init : npt.NDArray[np.float64]
        Initial positions of the beads.
    pos_ind_preload : npt.NDArray[np.float64]
        Preload positions for the beads.
    beads : list[int]
        Indices of the beads to optimize.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    trajectory[npt.NDArray[np.float64]]
        Trajectory of the bead positions
    """
    response_options = {PROGRESS_BAR: True}
    response_options.update({k: v for k, v in kwargs.items() if k in response_options})

    preload_steps = pos_ind_preload.shape[0]
    laplacian = kwargs.get("laplacian")
    assert isinstance(laplacian, np.ndarray)

    pos_now = pos_init.copy()
    trajectory = [pos_init]

    iter_preloads = pbar(range(preload_steps), response_options[PROGRESS_BAR])

    for i in iter_preloads:
        pos_preload = pos_now.copy()
        pos_preload[[*beads]] = pos_ind_preload[i]

        # Calculate the Hessian matrix
        if kwargs["keep_eq_dist"]:
            hessian = calc_hessian_with_nzeqd(kwargs["eq_dist"], pos_now, kwargs["connected"])
        else:
            hessian, _, _ = calc_hessian(pos_now, laplacian)

        if kwargs["repulsive_beads"]:
            hessian += repulsive_hessian(
                pos_now,
                kwargs["repulsive_cutoff"],
                kwargs["mapping"],
                kwargs["epsilon"],
                kwargs["sigma"],
            )

        # perform the optimization step
        pos_now = simple_qopt(
            pos_now,
            pos_preload,
            hessian,
            beads,
            suppress_NullSpaceError=kwargs.get("suppress_NullSpaceError", False),
        )
        trajectory.append(pos_now)

    trajectory = np.array(trajectory)
    return trajectory


def concentric(
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """
    Calculate the response using the simplified 'QOPT' method,
    pulling the beads towards their shared center of mass/gravity.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments for the QOPT calculation.

    Returns
    -------
    trajectory[npt.NDArray[np.float64]]
        Trajectory of the bead positions

    default options are:
            default_options = {
            "optimizer": optimizer,
            "bead_positions": network.geometry.bead_positions,
            "max_pull": network.max_pull,
            "beads": network.source,
            "adaptive_connections": adaptive_connections,
            "laplacian": network.geometry.laplacian,
            "swap_beads": False,
            "repulsive_beads": False,
            "keep_eq_dist": True,
            "eq_dist": network.geometry.eq_dist,
            "connected": network.geometry.connected,
            "suppress_NullSpaceError": False,
            "mapping": network.dense_square_map,
            "epsilon": network.geometry.epsilon,
            "sigma": network.geometry.sigma,
            "repulsive_cutoff": network.geometry.repulsive_cutoff,
            "cutoff_length": network.geometry.cutoff_length,
            "noise_strength": 0.0,
            "preload_steps": 50,
        }
    """
    preload_steps = kwargs.get("preload_steps", 50)
    bead_pos = kwargs["bead_positions"]
    max_pull = kwargs["max_pull"]
    beads = kwargs["beads"]
    swap_beads = kwargs["swap_beads"]
    variant = kwargs.get("variant")
    noise_strength = kwargs.get("noise_strength", 0.0)

    # Calculate preload positions
    if swap_beads:
        if variant == "antisymmetric":
            preloads = np.linspace(0, -(1 - max_pull), preload_steps + 1)[1:]
        elif variant == "symmetric":
            preloads = np.linspace(0, 1 - max_pull, preload_steps + 1)[1:]
    else:
        preloads = np.linspace(0, 1 - max_pull, preload_steps + 1)[1:]

    pos_ind_preload = preload_positions(bead_pos, beads, preloads, noise_strength=noise_strength)

    kwargs.pop("beads", None)
    return _base_qopt(bead_pos, pos_ind_preload, beads, **kwargs)


def guided(
    pos_ind_preload: npt.NDArray[np.float64],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """
    Calculate the response using the simplified 'QOPT' method,
    pulling the beads along predefined preload positions (trajectories).

    Parameters
    ----------
    pos_ind_preload : npt.NDArray[np.float64]
        Predefined preload positions for the beads.
    **kwargs : Any
        Additional keyword arguments for the QOPT calculation.

    Returns
    -------
    trajectory[npt.NDArray[np.float64]]
        Trajectory of the bead positions

    default options are:
        default_options = {
            "optimizer": optimizer,
            "bead_positions": network.geometry.bead_positions,
            "max_pull": network.max_pull,
            "beads": network.source,
            "adaptive_connections": adaptive_connections,
            "laplacian": network.geometry.laplacian,
            "swap_beads": False,
            "repulsive_beads": False,
            "keep_eq_dist": True,
            "eq_dist": network.geometry.eq_dist,
            "connected": network.geometry.connected,
            "suppress_NullSpaceError": False,
            "mapping": network.dense_square_map,
            "epsilon": network.geometry.epsilon,
            "sigma": network.geometry.sigma,
            "repulsive_cutoff": network.geometry.repulsive_cutoff,
            "cutoff_length": network.geometry.cutoff_length,
            "noise_strength": 0.0,
            "preload_steps": 50,
        }
    """
    bead_pos = kwargs["bead_positions"]
    beads = kwargs["beads"]

    kwargs.pop("beads", None)

    return _base_qopt(bead_pos, pos_ind_preload, beads, **kwargs)
