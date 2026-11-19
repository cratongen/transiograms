class TransiogramsError(Exception):
    """Base class of every error raised by transiograms."""


class InvalidInput(TransiogramsError, ValueError):
    """An argument is invalid; the message names the offending item."""


class ConvergenceError(TransiogramsError, RuntimeError):
    """An iterative method did not converge or a decomposition failed."""
