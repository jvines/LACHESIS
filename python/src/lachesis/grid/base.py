"""Abstract grid interface — every isochrone grid implements this."""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class IsochroneGrid(Protocol):
    """Protocol for isochrone grids (MIST, PARSEC, Dartmouth, ...)."""

    @property
    def name(self) -> str: ...

    @property
    def eep_range(self) -> tuple[int, int]: ...

    @property
    def feh_values(self) -> NDArray[np.float64]: ...

    @property
    def age_values(self) -> NDArray[np.float64]: ...

    @property
    def columns(self) -> list[str]: ...

    def __call__(
        self, eep: float, log_age: float, feh: float
    ) -> dict[str, float]:
        """Interpolate grid at (EEP, log_age, [Fe/H]) → observables."""
        ...
