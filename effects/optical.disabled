import numpy as np
from numpy.typing import ArrayLike

from .base import Effect


class OpticalEffect(Effect):
    """
    Effects and problems affecting the optical path.
    Should be formulated in terms of (r, b).
        r: radial distance from the optical path [mm]
        b: azimuth
    """

    def __call__(self,
                 r: ArrayLike, # Radial distance from the optical axis,
                 b: ArrayLike, # Azimuth, relative to the optical axis
                 ) -> ArrayLike:
        return np.zeros(size=(r, b))


class Vignetting(OpticalEffect):
    def __init__(self,
                 func: Callable[[ArrayLike], ArrayLike]
        """
        func: radial function
        """
        self.func = func

    def __call__(self,
                 r: ArrayLike,
                 b: ArrayLike) -> ArrayLike:
        dimming = np.ones(shape=(r, b))
        return dimming * original
