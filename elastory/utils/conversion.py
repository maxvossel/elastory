##############################
# convert cartesian to polar #
##############################

import numpy as np
from numba import njit


@njit
def polar_to_cartesian(
    r: float,
    theta: float,
    phi: float,
) -> tuple[float, float, float]:
    """
    Convert from polar coordinates to cartesian coordinates.
    Args:
        r: float
            Radius
        theta: float
            Angle
        phi: float
            Angle
    Returns:
        x, y, z: float
            Cartesian coordinates
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    x = r * st * cp
    y = r * st * sp
    z = r * ct
    return x, y, z


@njit
def cartesian_to_polar(
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    """
    Convert from cartesian coordinates to polar coordinates.
    Args:
        x, y, z: float
            Cartesian coordinates
    Returns:
        r, theta, phi: float
            Polar coordinates
    """
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi
