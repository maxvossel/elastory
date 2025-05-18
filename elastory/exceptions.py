class OptError(Exception):
    """Base class for excetions in the optimizer routines"""


class NullSpaceError(OptError):
    """Raised when the nullspace dimension is too big"""


class ConvergenceError(OptError):
    """Raised when the linear system did not yield a correct solution"""


class ResponseError(Exception):
    """Base class for excetions in the routine calculating the response"""


class BeadsCollideError(ResponseError):
    """Raised when some beads are too close"""


class VariantError(Exception):
    """Raised when the snet has no variant specified"""
