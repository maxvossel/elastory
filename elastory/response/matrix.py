import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvalsh, null_space
from scipy.sparse.linalg import eigsh

from elastory.exceptions import NullSpaceError
from elastory.utils.misc import degrees_of_freedom


def validate_dimensions(mat: npt.NDArray[np.float64]):
    if len(mat.shape) > 2:
        raise ValueError(f"Only tested for 1 or 2 dimensional arrays, you gave: {len(mat.shape)}")
    if (len(mat.shape) == 2) and (mat.shape[0] != mat.shape[1]):
        raise ValueError(f"Works only for square arrays, you gave: {mat.shape}")


def order_blocks(
    mat: npt.NDArray[np.float64],
    inds: tuple[int],
    D: int = 3,
) -> npt.NDArray[np.float64]:
    """
    Sort rows and columns in a np.ndarray (square if 2 dim) according
    to the following scheme: For each of the given indices,
    subsequently swap its correspondig D columns (and rows) to the
    front of the array.
    Args:
        mat  : (N*D[, N*D]) np.ndarray, matrix where N is the
               number of beads, corresponding to the indices
        inds : list, indices which to gather
        D    : int, spatial dimension of the underlying system
    Returns:
        m   : (N*D[, N*D]) np.ndarray, sorted matrix
    """

    validate_dimensions(mat)

    m = mat.copy()
    c = len(inds)
    origs = [*inds]
    dests = [*range(c)]
    never_move = dests[:]

    for ind in inds:
        if ind in never_move:
            origs.remove(ind)
            dests.remove(ind)

    for dest, orig in zip(dests, origs):
        for d in range(D):
            f = D * orig + d
            t = D * dest + d

            if len(mat.shape) == 1:
                m[[t, f]] = m[[f, t]]
            elif len(mat.shape) == 2:
                m[[t, f], :] = m[[f, t], :]
                m[:, [t, f]] = m[:, [f, t]]

    return m


def cut_blocks(
    mat: npt.NDArray[np.float64],
    c: int,
    D: int = 3,
    symmetric: bool = False,
) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Cut an array in two (if 2 dimensional four (three if symmetric, respectively))
    blocks according to the indices given.
    Args:
        mat       : (N*D[, N*D]) np.ndarray, matrix where N is the
                    number of beads, corresponding to the indices
        c         : index at which to cut
        D         : int, spatial dimension of the underlying system
        symmetric : whether mat is equal to mat.T
    Returns:
        m_i       : (N*D[, N*D]) np.ndarray, blocks of the input array
    """

    validate_dimensions(mat)

    m = mat.copy()
    c *= D

    match len(mat.shape):
        case 1:
            m1 = m[:c]
            m2 = m[c:]
            return m1, m2
        case 2:
            m1 = m[:c, :c]
            m2 = m[:c, c:]
            m3 = m[c:, :c]
            m4 = m[c:, c:]
            return (m1, m2.T, m4) if symmetric else (m1, m2, m3, m4)
        case _:
            raise ValueError(
                f"Only tested for arrays of dimension 1 or 2, you gave: {len(mat.shape)}"
            )


def remove_nullspace(
    mat: npt.NDArray[np.float64],
    D: int = 3,
    rcond: float = 1e-10,
    suppress_NullSpaceError: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Removes the nullspace of the matrix while assuring that the
    dimension of the nullspace is never bigger than the degrees
    of freedom of the underlying rigid body.
    Args:
        mat : (N*D, N*D) ndarray, matrix
        D   : int, spatial dimension
    Returns:
        mat : the matrix without nullspace
    """
    dof = degrees_of_freedom(D)
    m = mat.copy()
    NS = null_space(mat, rcond=rcond)

    # TODO: I still have to test this part -> it should in principle
    # be equivalent to the scipy version from above but faster
    # vs = fast_nullspace_basis(mat)
    # NS = np.outer(vs, vs)

    # Get only the eigenvalue at index ind_last_w
    ind_last_w = mat.shape[0] - 1
    w_max: npt.NDArray[np.float64] = eigvalsh(mat, subset_by_index=[ind_last_w, ind_last_w])[0]
    # for an older scipy version, the code was:
    # `w_max = eigvalsh(mat, eigvals=(ind_last_w, ind_last_w))[0]`

    if NS.shape[1] == 0:
        return m

    if (NS.shape[1] > dof) and not suppress_NullSpaceError:
        raise NullSpaceError(
            f"NS dimension ({NS.shape[1]}) is bigger than the expected maximum ({dof}),"
            f"the number of degrees of freedom associated with rigid body motions in a "
            f"spatial dimension of {D=}."
        )

    if NS.shape[1] == 1:
        m += w_max * np.kron(NS.T, NS)

    if NS.shape[1] > 1:
        for ns in NS.T:
            ns = ns.reshape((1, -1))
            m += w_max * np.kron(ns, ns.T)

    assert isinstance(m, np.ndarray)
    return m


def fast_nullspace_basis(
    M: np.ndarray,
    eps: float = 1e-9,
    D: int = 3,
    suppress_NullSpaceError: bool = False,
) -> np.ndarray:
    """
    Using shift invert and the arpack iterative method for
    finding extremal eigenvalues this method returns the
    nullspace of the matrix M. Overcoming a bug by adding
    a bit of small noise with magnitude eps.
    Args:
        M   : (N*D, N*D) ndarray, matrix
        eps : (N*D, N*D) ndarray, matrix
        D   : int, spatial dimension of the underlying system
    Returns:
        vs : (N*D,) ndarray,
             the eigenvectors spanning the nullspace
        ws : (N*D,) ndarray,
           : the eigenvalues (should all be ~eps)
    """
    dof = degrees_of_freedom(D)
    r = np.random.rand(*M.shape) * eps
    M_r = M + r
    ws, vs = eigsh(M_r, k=dof, sigma=eps)

    assert isinstance(ws, np.ndarray)
    assert isinstance(vs, np.ndarray)

    if not suppress_NullSpaceError:
        if vs.shape[1] > dof:
            raise NullSpaceError(
                f"NS dimension ({vs.shape[1]}) is bigger than the expected maximum ({dof}), \n "
                + "which is the number of degrees of freedom associated with rigid "
                + "body motions of a body in ({D}) spatial dimensions."
                ""
            )

    return vs


def remove_rigid_body_motions(
    hessian: npt.NDArray[np.float64],
    D: int = 3,
    scale: float = 1e3,
) -> npt.NDArray[np.float64]:
    """
    Changes the eigenvalues which are (numerically) zero to
    high values to suppress the corresponding motion, using a
    spectral decomposition of the hessian matrix.
    Args:
        hessian : (N*D[, N*D]) np.ndarray, hessian where N is the
                   number of beads, corresponding to the indices
        D        : int, spatial dimension of the underlying system
        scale    : float, scale factor for the suppression of the
                   rigid body motions
    Returns:
        hessian_no_rbm : (N*D[, N*D]) np.ndarray, hessian matrix with no
                         rigid body motions encoded
    """
    dof = degrees_of_freedom(D)
    hessian_no_rbm = hessian.copy()
    w_suppress = scale + np.arange(dof)
    v_rb = fast_nullspace_basis(hessian)

    for n, v in enumerate(v_rb.T):
        hessian_no_rbm += np.outer(v, v) * w_suppress[n]

    return hessian_no_rbm
