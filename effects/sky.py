import numpy as np

from abc import abstractmethod

from typing import Union, Optional

from demeteor.metrics import spherical
from numpy.typing import ArrayLike

from astropy.coordinates import EarthLocation, get_body, AltAz
from astropy.time import Time
import astropy.units as u


class SkyEffect:
    """
    Effects defined in terms of angles in the sky.
    In general can be purely additive (sources, such as stars, scattered light, airglow)
    or subtractive / dividing (clouds, obstructions, ...).
    """
    def __init__(self,
                 location: EarthLocation,
                 time: Optional[Time] = None):
        self.location = location
        self.time = time if time is not None else Time.now()
        self.altaz = AltAz(obstime=self.time, location=self.location)

    @abstractmethod
    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        """
        Transform the value at data by the corresponding value at (`alt`, `az`)
        """


class SkySource(SkyEffect):
    """
    SkySource is a source -- its function is purely additive.
    """
    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        return data + self.func(alt, az)

    @abstractmethod
    def func(self,
             alt: ArrayLike,
             az: ArrayLike) -> ArrayLike:
        """
        The inner function of the source, defined at (`alt`, `az`)
        """

    @staticmethod
    def path_length(alt: ArrayLike) -> ArrayLike:
        """
        Path length from infinity at altitude `alt`.
        """
        return (1 - 0.96 * np.cos(alt)) ** -0.5


class Sunlight(SkySource):
    def func(self,
             alt: ArrayLike,
             az: ArrayLike) -> ArrayLike:
        sun = get_body('sun', self.time, self.location)
        sun = sun.transform_to(self.altaz)
        brightness = -26.74 * u.mag

        pixels = np.stack((alt, az), axis=2)
        dist = spherical(pixels, np.stack((sun.alt.radian, sun.az.radian), axis=0))

        intensity = 10 * (0.8 * np.sin(sun.alt) + 0.2) if sun.alt >= 0 else 0.1 * np.exp(sun.alt.radian * 10)

        return np.where(
            alt <= 0,
            0,
            25 * np.exp(-alt * 5) * np.exp(-dist**2) * intensity
        )


class Airglow(SkySource):
    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        super().__init__(location, time)
        self.intensity = kwargs.pop('intensity', 0)

    def func(self,
             alt: ArrayLike,
             az: ArrayLike) -> ArrayLike:
        x = self.path_length(alt)
        extinction = 0.145
        return np.where(
            alt <= 0, 0,
            self.intensity * 10**(-0.4 * extinction * (x - 1)) * x,
        )


class Extinction(SkyEffect):
    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        super().__init__(location, time)
        self.k = kwargs.pop('k', 0.145)

    @staticmethod
    def path_length(alt: ArrayLike) -> ArrayLike:
        """
        Path length from infinity at altitude `alt`.
        """
        u = np.sin(alt)
        return np.where(
            alt > 0,
            # 1 / np.sin(alt + 244 / (165 + 47 * np.degrees(alt)**1.1)), Pickering 2002 is useless shit
            (1.002432 * u**2 + 0.148386 * u + 0.0096467) / (u**3 + 0.149846 * u**2 + 0.0102963 * u + 0.000303978),
            1000
        )

    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        x = self.path_length(alt)
        return data * 10**(-0.4 * self.k * x)


class Moonlight(SkySource):
    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        super().__init__(location, time)
        self.extinction = kwargs.pop('extinction', 0)

    @staticmethod
    def separated(ang: ArrayLike) -> ArrayLike:
        return 10**5.36 * (1.06 + np.cos(ang)**2) + 10**(6.15 - np.degrees(ang) / 40)


    @staticmethod
    def intensity(phase: ArrayLike) -> ArrayLike:
        return 10**(-0.4 * (3.84 + 0.026 * np.abs(np.degrees(phase)) + 4e-9 * np.degrees(phase)**4))

    def brightness(self, alt: ArrayLike) -> ArrayLike:
        """
        alt: altitude
        """
        return 10**(-0.4 * self.extinction * self.path_length(alt))

    def func(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        moon = get_body('moon', self.time, self.location)
        sun = get_body('sun', self.time, self.location)
        phase = moon.separation(sun)

        moon_altaz = moon.transform_to(self.altaz)

        pixels = np.stack((alt, az), axis=2)
        dist = spherical(pixels, np.stack((moon_altaz.alt.radian, moon_altaz.az.radian), axis=0).T)

        return self.separated(dist) * self.intensity(phase.value) * self.brightness(np.pi / 2 - moon_altaz.alt.radian) * (1 - self.brightness(np.pi / 2 - alt))