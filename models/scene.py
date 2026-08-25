import logging
from typing import Callable

import matplotlib as mpl
import numpy as np
from demeteor.catalogue import Catalogue
from demeteor.projections.shifters import ScalingShifter
from numpy.typing import ArrayLike
from PIL import Image

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.projections import BorovickaProjection, Projection

from effects.sky import SkySource, Sunlight, Airglow, Moonlight, Extinction
from models.skypointsource import SkyPointSource

u.Wm2 = u.W / u.m**2
u.ms = u.m / u.s


class Scene:
    gain = 1e13 / u.Wm2

    def __init__(self,
                 xres: int,
                 yres: int,
                 projection: Projection,
                 scaler: ScalingShifter,
                 location: EarthLocation,
                 catalogue: Catalogue,
                 time: Time = None):
        self.xres = xres
        self.yres = yres
        self.data = np.zeros(shape=(yres, xres))
        self.xs, self.ys = np.meshgrid(np.arange(0, xres), np.arange(0, yres))

        self.location = location
        self.time = Time.now() if time is None else time
        self.projection = projection
        self.scaler = scaler
        self.catalogue = catalogue

        # Pre-compute sky altitude and azimuth for every pixel in the scene
        self.alt, self.az = self.projection(*self.scaler(self.xs, self.ys))
        self.alt = np.pi / 2 - self.alt

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

    def build(self, fragments: list[SkyPointSource]):
        logging.info(f"Building a scene ({self.xres}x{self.yres}) at {self.time}")

        sun = Sunlight(self.location, self.time)
        moon = Moonlight(self.location, self.time)
        airglow = Airglow(self.location, self.time, intensity=100)
        extinction = Extinction(self.location, self.time)
        self.add_stars()
        self.add_fragments(fragments)
        self.add_sky_effects([sun, moon, extinction, airglow])

    def render(self, filename = None):
        logging.info(f"Rendering the scene to file {filename} {self.data.T.shape}")
        readout = self.flux_to_electrons(lambda x: x * 10)
        readout = np.flip(self.rescale(readout), axis=0)
        Image.fromarray(readout).save(filename)

    def add_stars(self):
        """
        Add stars as defined by the catalogue
        """
        altaz = self.catalogue.altaz(self.location, self.time, masked=False)
        mask = altaz.alt > 0
        self.catalogue.mask = mask
        altaz = altaz[mask]

        # The time, and not just the location. Catalogue.vmag() rebuilds the planets from scratch
        # -- positions and brightnesses both -- and with no time it rebuilds them for whenever this
        # happens to be running: seven bodies whose alt-az was computed for the simulated epoch two
        # lines above, given the magnitude they have today. Mars alone runs from -2.9 to +1.8, a
        # factor of a hundred in flux.
        ints = self.vmag_to_intensity(self.catalogue.vmag(self.location, self.time, masked=True))
        self.add_points(altaz.alt, altaz.az, ints)

    def add_sky_effects(self,
                        sources: list[SkySource]):
        for source in sources:
            self.data = source(self.data, self.alt, self.az)

    def add_fragments(self, fragments: list[SkyPointSource]):
        for fragment in fragments:
            alt, az, inten = fragment.at_time(self.time)
            self.add_points(alt, az, inten)

    @staticmethod
    def vmag_to_intensity(vmag: ArrayLike) -> u.Quantity[u.W / u.m ** 2]:
        return np.exp(-0.921034 * (vmag + 19.89)) * u.W / u.m ** 2

    @staticmethod
    def intensity_to_sigma(intensity: u.Quantity) -> ArrayLike:
        """
        Obtain the Gaussian spatial profile variance as a function of brightness.
        Some highly non-trivial function is expected here.
        """
        return np.where(
            intensity < 1e-12 * u.Wm2,
            0.5,
            np.maximum(0.5, (np.log2(intensity / u.Wm2) + 27)**3 / 40)
        )

    def add_points(self,
                   alt: ArrayLike,
                   az: ArrayLike,
                   intensities: ArrayLike) -> None:
        """
        Add a collection of point sources (defined by `alt`, `az`) to the scene.
        """
        assert alt.shape == az.shape == intensities.shape, \
            f"Coordinates and intensities have a wrong shape: {alt.shape=}, {az.shape=}, {intensities.shape=}"

        alt = alt.to(u.rad).value
        az = az.to(u.rad).value
        # Obtain coordinates in the detector coordinates
        mx, my = self.projection.invert(np.pi / 2 - alt, az)
        x, y = self.scaler.invert(mx, my)

        mask = (x >= 0) & (x < self.xres) & (y >= 0) & (y < self.yres)
        nx, ny = x[mask], y[mask]
        intensities = intensities[mask]
        sigmas = self.intensity_to_sigma(intensities)
        truncate = 5.0

        # ChatGPT: Loop over deltas, but vectorized over each patch
        for xi, yi, ii, si in zip(nx, ny, intensities, sigmas):
            rad = int(np.ceil(truncate * si) + 1)
            xmin, xmax = max(0, int(np.floor(xi)) - rad), min(self.xres, int(np.floor(xi)) + rad + 1)
            ymin, ymax = max(0, int(np.floor(yi)) - rad), min(self.yres, int(np.floor(yi)) + rad + 1)

            # Local patch grid
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            g = np.exp(-((xx - xi)**2 + (yy - yi)**2) / (2 * si**2))

            # Normalize and scale
            g /= g.sum()
            g *= self.gain * ii

            # Add the patch to result
            self.data[ymin:ymax, xmin:xmax] += g

    def add_intensifier_noise(self,
                              rate: float = 100,
                              multiplier: float = 10,
                              *,
                              radius: float = 50):
        """
        A function to simulate AMOS optical intensifier noise in the centre of field
        """
        noise = np.random.poisson(rate, size=(self.yres, self.xres))
        xc, yc = self.xres / 2, self.yres / 2
        profile = np.exp(-((self.xs - xc)**2 + (self.ys - yc)**2) / (2 * radius**2))
        self.data += multiplier * noise * profile

    def add_thermal_noise(self, rate=40):
        """
        Add thermal noise to the entire area of the detector. Very basic, currently Poisson.
        """
        noise = 20 * np.random.poisson(rate, size=(self.yres, self.xres))
        self.data += noise

    def render_as_poisson(self):
        """
        Translate intensity into Poisson noise: treat incoming light intensity as
        the rate of a Poisson distribution and emulate shot noise.
        """
        self.data = np.random.poisson(self.data, size=(self.yres, self.xres)).astype(np.float64)

    def flux_to_electrons(self, func: Callable[[ArrayLike], ArrayLike]) -> ArrayLike:
        """
        Transform physical flux into electrons or ADU.
        """
        return func(self.data)

    @staticmethod
    def rescale(readout: ArrayLike, max_adu: int = 65535) -> ArrayLike:
        """
        Rescale the detector readout to [0, 255], in order to save to a regular bitmap.

        Parameters
        ----------
        readout: ArrayLike
            Final detector readout (#ToDo currently)
        max_adu: int
            ADU that correspond to detector saturation

        Returns
        -------
        ArrayLike
            A rescaled detector readout
        """
        return (np.clip(readout, 0, max_adu) / max_adu * 255).astype(np.uint8)
