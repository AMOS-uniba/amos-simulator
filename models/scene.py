import logging
from typing import Callable

import matplotlib as mpl
import numpy as np
from demeteor.catalogue import Catalogue
from demeteor.projections.shifters import ScalingShifter
from numpy.typing import ArrayLike
from PIL import Image
from scipy.special import erf

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.projections import BorovickaProjection, Projection

from effects import airmass, brightness
from effects.sky import Emission, Extinction, Sunlight, Airglow, Moonlight
from models.detector import Detector
from models.skypointsource import SkyPointSource

u.Wm2 = u.W / u.m**2
u.ms = u.m / u.s


class Scene:

    def __init__(self,
                 xres: int,
                 yres: int,
                 projection: Projection,
                 scaler: ScalingShifter,
                 location: EarthLocation,
                 catalogue: Catalogue,
                 time: Time = None,
                 sky: dict = None,
                 detector: Detector = None,
                 psf: dict = None,
                 subsamples: int = 1):
        # What the atmosphere is like tonight. A dict rather than a DotMap because this is handed to
        # a Pool worker and has to pickle; `config/renderers/*.yaml` is where the numbers live.
        self.sky = dict(sky or {})
        # How the flux in each pixel becomes a number. Default so that a caller building a scene by
        # hand -- a test, a notebook -- gets a working camera rather than a required argument.
        self.detector = detector if detector is not None else Detector()
        # The point spread function, as a width in the middle of the field and one at the rim.
        psf = dict(psf or {})
        self.sigma_centre = psf.get('fwhm_centre', 2.4) / 2.3548
        self.sigma_edge = psf.get('fwhm_edge', 3.6) / 2.3548
        self.subsamples = int(subsamples)
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
        """
        Fill the frame: what came through the atmosphere, then what the atmosphere added.

        The order is the physics and is no longer a property of a list. Stars, planets and the meteor
        arrived from outside, so they are summed and then dimmed together -- one column, one air mass
        per pixel. Airglow, scattered moonlight and twilight are made inside it and are added after,
        each carrying its own geometry.
        """
        logging.info(f"Building a scene ({self.xres}x{self.yres}) at {self.time}")

        self.add_stars()
        self.add_fragments(fragments, exposure=self.detector.exposure * u.s,
                           subsamples=self.subsamples)
        self.attenuate()
        self.add_emission()

    def attenuate(self) -> None:
        """ Dim everything gathered so far by the air it came through. """
        extinction = Extinction(self.location, self.time,
                                extinction=self.sky.get('extinction', airmass.EXTINCTION))
        self.data = extinction(self.data, self.alt, self.az)

    def pixel_solid_angle(self) -> float:
        """
        How much sky one pixel sees at the centre of the field, in square arcseconds.

        Measured off the projection rather than assumed, by asking where the pixel next door looks.
        It is the centre value and the field is not uniform -- a fisheye pixel near the horizon covers
        several times more sky than one at the zenith -- so a surface brightness converted with this
        is right in the middle and increasingly wrong towards the rim. Doing that properly means the
        projection's Jacobian at every pixel; this is the honest one-number version, and the error it
        carries is a smooth radial one that no fit of a plate would notice.
        """
        centre = np.array([self.xres / 2.0, self.yres / 2.0])
        r0, a0 = self.projection(*self.scaler(*centre))
        r1, a1 = self.projection(*self.scaler(*(centre + np.array([1.0, 0.0]))))
        r2, a2 = self.projection(*self.scaler(*(centre + np.array([0.0, 1.0]))))
        # Two orthogonal steps of one pixel, in radians of sky, times each other
        step_x = np.hypot(r1 - r0, (a1 - a0) * np.sin(r0))
        step_y = np.hypot(r2 - r0, (a2 - a0) * np.sin(r0))
        return float(np.degrees(step_x) * np.degrees(step_y) * 3600.0 ** 2)

    def emissions(self) -> list[Emission]:
        """ The atmosphere's own light, as configured. Light pollution will join this list. """
        k = self.sky.get('extinction', airmass.EXTINCTION)
        glow = self.sky.get('airglow', {})
        common = dict(extinction=k, pixel_solid_angle=self.pixel_solid_angle())
        sources: list[Emission] = [
            Airglow(self.location, self.time, **common,
                    brightness=glow.get('brightness', 21.8),
                    height=glow.get('height', airmass.AIRGLOW_HEIGHT)),
        ]
        if self.sky.get('moon', True):
            sources.append(Moonlight(self.location, self.time, **common))
        # Off by default, and the class says why: its constants are in no unit.
        if self.sky.get('sun', False):
            sources.append(Sunlight(self.location, self.time, **common))
        return sources

    def add_emission(self) -> None:
        for source in self.emissions():
            self.data = source(self.data, self.alt, self.az)

    def render(self, filename=None):
        """
        Take the accumulated flux through the camera and write the frame.

        The flip is display, not physics: a sensor numbers its rows downwards and an image file is
        read the same way, while `alt` grows upwards.
        """
        logging.info(f"Rendering the scene to file {filename} {self.data.T.shape}")
        readout = self.detector.readout(self.data, self.xs, self.ys)
        Image.fromarray(np.flip(readout, axis=0).astype(np.uint8)).save(filename)

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

    def add_fragments(self, fragments: list[SkyPointSource], exposure=None, subsamples: int = 1):
        """
        Draw the meteor as it was over the whole exposure, not as it was at one instant.

        A meteor crosses tens of pixels while the shutter is open, so a single sample per frame
        renders a dot where a camera records a streak -- and a streak is what Kvant measures, both
        for position and for brightness. Each of `subsamples` sub-times carries its share of the
        flux, so the total light in the frame is unchanged and only its distribution moves.

        `SkyPointSource.at_time` interpolates to any instant and returns zero outside the flight, so
        a sub-time that falls before the meteor started contributes nothing, correctly.
        """
        if exposure is None or subsamples <= 1:
            offsets = [0.0 * u.s]
        else:
            # Centres of `subsamples` equal slices of the exposure, which is centred on self.time
            step = exposure / subsamples
            offsets = (np.arange(subsamples) - (subsamples - 1) / 2.0) * step

        weight = 1.0 / len(offsets)
        for fragment in fragments:
            for offset in offsets:
                alt, az, inten = fragment.at_time(self.time + offset)
                self.add_points(alt, az, inten * weight)

    @staticmethod
    def vmag_to_intensity(vmag: ArrayLike) -> u.Quantity[u.W / u.m ** 2]:
        """ One zero point, in effects/brightness.py, for stars and sky brightness alike. """
        return brightness.flux_from_magnitude(vmag) * u.W / u.m ** 2

    def psf_sigma(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        The Gaussian width to draw a point source with, in pixels, at position (`x`, `y`).

        Two numbers: the width in the middle of the field and the width at the edge of the circle,
        interpolated linearly in radius. Nothing fancier is warranted -- at these field angles a real
        all-sky lens is soft towards the rim and that is all this is meant to say -- but the floor
        matters: the width this replaces bottomed out at half a pixel, which puts a faint star inside
        a single pixel, where its centroid is decided by the pixel grid rather than by the light.

        What is gone with it is the old brightness dependence, `(log2(I) + 27)**3 / 40`, which was
        standing in for the way a bright star spreads. That is saturation, not optics, and the
        detector's full well now produces it for the right reason: a flat-topped core is a clipped
        core.
        """
        radius = min(self.xres, self.yres) / 2.0
        distance = np.hypot(np.asarray(x) - self.xres / 2.0, np.asarray(y) - self.yres / 2.0)
        fraction = np.clip(distance / radius, 0.0, 1.0)
        return (self.sigma_centre + (self.sigma_edge - self.sigma_centre) * fraction)

    @staticmethod
    def pixel_gaussian(centre: float, low: int, high: int, sigma: float) -> ArrayLike:
        """
        A Gaussian integrated over each pixel from `low` to `high`, rather than sampled at centres.

        Separable, so a patch is the outer product of two of these, and honest at small sigma: a
        pixel holds what fell on it, not the value of the profile at its middle. Sampling instead
        biases the centroid of a sigma = 0.5 px source by up to 0.023 px, which is small but is a
        bias rather than a scatter, and it costs nothing to not have it.
        """
        # Pixel *boundaries*, so n pixels need n + 1 of them: from low - 0.5 to high + 0.5
        edges = (np.arange(low, high + 2) - 0.5 - centre) / (sigma * np.sqrt(2.0))
        return 0.5 * np.diff(erf(edges))

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
        # `data` holds a flux per pixel in W/m2 and carries no unit of its own, so the unit comes off
        # here, once, where a source enters the frame. It used to come off by accident: the old
        # `gain = 1e13 / u.Wm2` multiplied it away on its way in, so removing the gain left a
        # quantity being added to a bare array.
        intensities = u.Quantity(intensities[mask]).to_value(u.W / u.m ** 2)
        sigmas = self.psf_sigma(nx, ny)
        truncate = 5.0

        for xi, yi, ii, si in zip(nx, ny, intensities, sigmas):
            rad = int(np.ceil(truncate * si) + 1)
            xmin, xmax = max(0, int(np.floor(xi)) - rad), min(self.xres, int(np.floor(xi)) + rad + 1)
            ymin, ymax = max(0, int(np.floor(yi)) - rad), min(self.yres, int(np.floor(yi)) + rad + 1)
            if xmin >= xmax or ymin >= ymax:
                continue

            # Separable, since a circular Gaussian is: the fraction of the source in each column
            # times the fraction in each row.
            gx = self.pixel_gaussian(xi, xmin, xmax - 1, si)
            gy = self.pixel_gaussian(yi, ymin, ymax - 1, si)
            g = np.outer(gy, gx)

            # Normalise to unit flux and give it the source's own -- normalised after truncation, so
            # a star at the very edge of the frame keeps all of the light that landed on the sensor.
            # What turns a flux into electrons is the detector's business, once, at readout.
            total = g.sum()
            if total <= 0:
                continue
            self.data[ymin:ymax, xmin:xmax] += g / total * ii
