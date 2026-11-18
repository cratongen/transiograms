class VarviteError(Exception):
    """Base class of every error raised by varvite."""


class InvalidInput(VarviteError, ValueError):
    """An argument is invalid; the message names the offending item."""


class ConvergenceError(VarviteError, RuntimeError):
    """An iterative method did not converge or a decomposition failed."""
