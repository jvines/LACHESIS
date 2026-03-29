"""LACHESIS exception hierarchy."""


class LachesisError(Exception):
    """Base exception for LACHESIS."""


class InputError(LachesisError):
    """Invalid user input (wrong types, missing required params)."""


class GridError(LachesisError):
    """Grid loading/interpolation failure."""


class PriorError(LachesisError):
    """Invalid prior configuration."""


class FittingError(LachesisError):
    """Nested sampling failure. Stores partial results for debugging."""

    def __init__(self, message, grid_name=None, partial_results=None):
        super().__init__(message)
        self.grid_name = grid_name
        self.partial_results = partial_results


class OutputError(LachesisError):
    """File I/O failure."""
