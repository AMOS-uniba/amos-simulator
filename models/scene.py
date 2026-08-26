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
from astropy.coordinates import AltAz, Angle, EarthLocation, get_body
from astropy.time import Time

from demeteor.projections import BorovickaProjection, Projection

from effects import airmass, brightness
from effects.sky import Emission, Extinction, Sunlight, Airglow, Moonlight
from models.detector import Detector
from models.skypointsource import SkyPointSource

u.Wm2 = u.W / u.m**2
u.ms = u.m / u.s

#: The Moon's radius in metres, which with its distance gives the angular size astropy does not
#: hand over directly.
MOON_RADIUS = 1737_400.0


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
        self.sigma_centre = psf.get('fwhm_centre', 1.8) / 2.3548
        self.sigma_edge = psf.get('fwhm_edge', 2.2) / 2.3548
        # The halo: a second, broad component carrying a few percent of the light. A Moffat, not a
        # Gaussian -- see halo_profile for why the shape of the wing is the whole point of it.
        self.halo_fraction = psf.get('halo_fraction', 0.03)
        self.halo_beta = psf.get('halo_beta', 1.8)
        self.halo_alpha = (psf.get('halo_fwhm', 12.0)
                           / (2.0 * np.sqrt(2.0 ** (1.0 / self.halo_beta) - 1.0)))
        self.halo_max = psf.get('halo_max', 200.0)
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
        if self.sky.get('moon', True):
            self.add_moon()
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
        if self.sky.get('sun', False):
            dusk = self.sky.get('twilight', {})
            sources.append(Sunlight(self.location, self.time, **common,
                                    brightness=dusk.get('brightness', 9.7),
                                    multiple_scattering=dusk.get('multiple_scattering', 0.4)))
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

    #: Sample the Moon's disc finely enough that the spacing is well inside a PSF width, or the disc
    #: comes out as a field of dots. Half a pixel, and bounded: at an all-sky plate scale the disc is
    #: four pixels across and a few hundred samples is already far more than enough, while a long lens
    #: makes it a hundred pixels and wants twenty thousand.
    MOON_SPACING = 0.5
    MOON_SAMPLES = (512, 40000)

    def add_moon(self, samples: int = None) -> None:
        """
        The Moon itself, as a disc rather than a dot.

        Its brightness is the same phase law the scattered-light model uses, with the zero point of
        the body instead of the one of Krisciunas & Schaefer's illuminance -- -12.73 at full, -9.25
        for the gibbous moon of 2025-10-01. Its size comes from its distance, which astropy hands
        over with the position: half a degree across, which at this plate scale is four pixels, so it
        is worth resolving.

        The disc is sampled on a sunflower spiral, which covers an area evenly without a preferred
        direction, and each sample carries its share of the flux through `add_points` -- so the Moon
        gets the same PSF as every star and the same extinction as everything else outside the
        atmosphere.

        **The phase is in the shape as well as in the brightness**: only the sunlit part of the disc
        is drawn, so the Moon is a crescent when it is one. On an all-sky plate this changes nothing
        anybody can see -- the terminator across four pixels is less than one, under a two pixel PSF,
        on something that saturates by five orders of magnitude -- and it is still worth having. The
        light of a crescent sits off the disc's centre by up to a third of a radius, which is the
        difference between the Moon's photometric centre and its geometric one, and a plate scale
        fine enough to see it is a config file away rather than a change here.
        """
        moon = get_body('moon', self.time, self.location).transform_to(
            AltAz(obstime=self.time, location=self.location))
        if moon.alt.radian <= 0:
            return

        # The Sun-Moon-Earth angle, which is the supplement of what `separation` gives -- see
        # Moonlight.phase_angle, and the bug it records.
        phase = Moonlight.phase_angle(self.location, self.time)
        flux = brightness.flux_from_magnitude(Moonlight.magnitude(phase))
        radius = np.arctan2(MOON_RADIUS, moon.distance.to(u.m).value)

        # How big it is on *this* plate, which decides how finely to sample it: project the centre
        # and a point one angular radius away and measure between them.
        mx, my = self.projection.invert(np.pi / 2 - moon.alt.radian, moon.az.radian)
        ex, ey = self.projection.invert(np.pi / 2 - moon.alt.radian - radius, moon.az.radian)
        pixels = float(np.hypot(*np.subtract(self.scaler.invert(ex, ey),
                                            self.scaler.invert(mx, my))))
        if samples is None:
            samples = int(np.clip((2.0 * pixels / self.MOON_SPACING) ** 2, *self.MOON_SAMPLES))

        # A sunflower spiral: r proportional to sqrt(k) spaces the samples by equal area, and the
        # golden angle keeps them from lining up into spokes.
        k = np.arange(samples) + 0.5
        r = radius * np.sqrt(k / samples)
        theta = k * np.pi * (3.0 - np.sqrt(5.0))

        # Offsets on the local tangent plane. Azimuth converges towards the zenith, hence the cosine;
        # the Moon is never near enough to it for that to be delicate.
        du, dv = r * np.cos(theta), r * np.sin(theta)
        alt = moon.alt.radian + du
        az = moon.az.radian + dv / np.cos(moon.alt.radian)

        # Which of those samples the Sun can see. In a frame whose first axis points along the sky
        # towards the Sun, a point (x, y) of the unit disc is lit when
        #
        #     x > -cos(phase) * sqrt(1 - y**2)
        #
        # -- the terminator projects to a half ellipse whose waist is cos(phase), positive towards
        # the Sun before quarter and away from it after, which is why one expression covers both the
        # gibbous and the crescent. It comes straight from n . s > 0 for the outward normal.
        sun = self.body_altaz('sun') if hasattr(self, 'body_altaz') else get_body(
            'sun', self.time, self.location).transform_to(
            AltAz(obstime=self.time, location=self.location))
        delta_az = sun.az.radian - moon.az.radian
        towards = np.array([
            np.sin(sun.alt.radian) * np.cos(moon.alt.radian)
            - np.cos(sun.alt.radian) * np.sin(moon.alt.radian) * np.cos(delta_az),
            np.cos(sun.alt.radian) * np.sin(delta_az),
        ])
        norm = np.hypot(*towards)
        if norm > 0:
            towards /= norm
            x = (du * towards[0] + dv * towards[1]) / radius
            y = (-du * towards[1] + dv * towards[0]) / radius
            lit = x > -np.cos(phase.to(u.rad).value) * np.sqrt(np.clip(1.0 - y ** 2, 0.0, 1.0))
        else:
            lit = np.ones_like(du, dtype=bool)

        if not lit.any():                       # a new Moon: nothing of it is turned this way
            return
        alt, az = alt[lit], az[lit]

        logging.info(f"The Moon: V = {Moonlight.magnitude(phase):.2f} at phase "
                     f"{phase.to(u.deg).value:.1f} deg, radius {np.degrees(radius) * 60:.1f} arcmin, "
                     f"{pixels * 2:.0f} px across, {lit.sum()}/{samples} of the disc sunlit")

        # The lit part of the disc without its halo, then the halo once, from the whole of its light.
        # The flux is shared between the samples that survived, so a crescent is not a dim full Moon:
        # the total is what the phase law says and it is spread over less of the sky.
        self.add_points(Angle(alt * u.rad), Angle(az * u.rad),
                        u.Quantity(np.full(alt.size,
                                           flux * (1.0 - self.halo_fraction) / alt.size),
                                   u.W / u.m ** 2), halo=False)
        self.add_halo(Angle(moon.alt.radian * u.rad), Angle(moon.az.radian * u.rad),
                      flux * u.W / u.m ** 2)

    def add_halo(self, alt: Angle, az: Angle, flux: u.Quantity) -> None:
        """
        The halo of one object, from the whole of its light, drawn once around a single point.

        For anything the sky presents as a point -- every star -- `add_points` does this itself. This
        is for an object sampled into many points, where the wing has to be added whole.
        """
        value = float(u.Quantity(flux).to_value(u.W / u.m ** 2))
        reach = self.halo_truncation(value)
        if reach <= 0:
            return

        mx, my = self.projection.invert(np.pi / 2 - alt.to(u.rad).value, az.to(u.rad).value)
        x, y = self.scaler.invert(mx, my)
        rad = int(np.ceil(reach) + 1)
        xmin, xmax = max(0, int(x) - rad), min(self.xres, int(x) + rad + 1)
        ymin, ymax = max(0, int(y) - rad), min(self.yres, int(y) + rad + 1)
        if xmin >= xmax or ymin >= ymax:
            return

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        self.data[ymin:ymax, xmin:xmax] += (self.halo_profile(np.hypot(xx - x, yy - y))
                                           * value * self.halo_fraction)

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

    #: How far out a profile is drawn at least, in units of sigma. Five puts 1e-6 of the flux
    #: outside, which is below anything a pixel can hold for an ordinary star.
    TRUNCATE = 5.0

    #: And at most -- a bound on the loop's cost, which in practice never binds. A Gaussian falls so
    #: steeply that brightness barely buys radius: the gibbous Moon asks for 5.8 sigma and the Sun,
    #: thirty million times brighter, for 8.1. **That is the finding, not the constant**: a factor of
    #: 3e7 in flux buys 1.4 in radius, so a saturated blob's size hardly depends on the source and
    #: bloom does not follow from brightness alone. A real halo is scattered light in the glass, with
    #: a power-law wing; modelling one means a second, broad PSF component, not a bigger number here.
    TRUNCATE_MAX = 20.0

    def halo_peak(self, flux: float) -> float:
        """ Surface brightness at the middle of the halo, from its analytic normalisation. """
        return flux * self.halo_fraction * (self.halo_beta - 1.0) / (np.pi * self.halo_alpha ** 2)

    def halo_profile(self, radius: ArrayLike) -> ArrayLike:
        """
        The wing: a Moffat, `(1 + r^2/a^2)^-beta`, normalised to one over the plane.

        A power law and not an exponential, which is the whole point. Fifty pixels out from something
        bright a Gaussian wing is 1e-21 of its peak and this is 2.5e-4 of it -- seventeen orders of
        magnitude -- and the second is what a bright source actually does to a frame. With a Gaussian
        halo the Moon's bloom stopped dead at 40 px; with this it reaches three hundred.

        `beta` is the knob: 1.5 for a lot of glow, 2.2 for a little.
        """
        alpha = self.halo_alpha
        return ((self.halo_beta - 1.0) / (np.pi * alpha ** 2)
                * (1.0 + (np.asarray(radius, dtype=float) / alpha) ** 2) ** -self.halo_beta)

    def halo_truncation(self, flux: float) -> float:
        """
        How far the halo is worth drawing, or zero when it cannot light a pixel anywhere.

        Inverting a power law costs a root where inverting a Gaussian costs a logarithm, which is why
        a bright source reaches so much further now. Bounded by `halo_max`, because a power law has no
        natural end and a patch has to be drawn.

        Faint stars never show one, so they keep a small patch and their three percent is normalised
        into the core -- three percent of the light moved a couple of pixels, inside any aperture
        anybody would measure with.
        """
        peak, floor = self.halo_peak(flux), self.detector.smallest_flux
        if not np.isfinite(peak) or peak <= floor:
            return 0.0
        reach = self.halo_alpha * np.sqrt((peak / floor) ** (1.0 / self.halo_beta) - 1.0)
        return float(min(reach, self.halo_max))

    def truncation(self, flux: float, sigma: float) -> float:
        """
        How far from a source to keep drawing it, in pixels.

        Not a constant times sigma. A saturated source is saturated out to wherever its profile last
        exceeds a pixel's smallest step, and that radius grows with brightness -- which is what makes
        the Moon a disc a dozen pixels across and a faint star three. With a fixed five sigma the size
        of the Moon's blob would be set by that constant rather than by the Moon.

        Still a Gaussian, so the edge is sharper than a real one: the halo around a real full Moon is
        scattered light in the glass and falls off as a power law. That wants a second, broad
        component in the PSF, and this is not it.
        """
        peak = flux / (2.0 * np.pi * sigma ** 2)
        floor = self.detector.smallest_flux
        if not np.isfinite(peak) or peak <= floor:
            return self.TRUNCATE * sigma
        return float(np.clip(sigma * np.sqrt(2.0 * np.log(peak / floor)),
                             self.TRUNCATE * sigma, self.TRUNCATE_MAX * sigma))

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
                   intensities: ArrayLike,
                   halo: bool = True) -> None:
        """
        Add a collection of point sources (defined by `alt`, `az`) to the scene.

        `halo=False` draws the core alone, for a caller sampling one extended object into many points
        that will add its halo once -- see add_moon. A halo belongs to the whole of an object's light,
        and cutting it into five hundred pieces truncates every piece early: the Moon's glow reached
        64 px that way, against the three hundred its own brightness asks for.
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

        for xi, yi, ii, si in zip(nx, ny, intensities, sigmas):
            rad = int(np.ceil(max(self.truncation(ii, si),
                                  self.halo_truncation(ii) if halo else 0.0)) + 1)
            xmin, xmax = max(0, int(np.floor(xi)) - rad), min(self.xres, int(np.floor(xi)) + rad + 1)
            ymin, ymax = max(0, int(np.floor(yi)) - rad), min(self.yres, int(np.floor(yi)) + rad + 1)
            if xmin >= xmax or ymin >= ymax:
                continue

            # Two components, both separable and both integrated over the pixel: a narrow core, and
            # a broad halo carrying `halo_fraction` of the light.
            #
            # Which is what a real star looks like. One Gaussian cannot be both sharp in the middle
            # and extended at the edges, so a single one has to be widened until its skirts look
            # right -- and then every star is hazy. Splitting them lets the core go down to the
            # sampling limit while the light that a real lens scatters into a glow stays in the
            # glow. It is also better photometry: a narrow core saturates a smaller area, and what
            # is outside an aperture is a fixed few percent rather than a function of the width.
            core = np.outer(self.pixel_gaussian(yi, ymin, ymax - 1, si),
                            self.pixel_gaussian(xi, xmin, xmax - 1, si))
            if halo and self.halo_fraction > 0:
                # Sampled at pixel centres rather than integrated over them, which the core cannot
                # afford and this does not need: at a dozen pixels across the wing hardly varies
                # inside one pixel, and it is not separable, so integrating it would cost a
                # quadrature per patch for no accuracy anybody could measure.
                yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
                wing = self.halo_profile(np.hypot(xx - xi, yy - yi))
                g = (1.0 - self.halo_fraction) * core + self.halo_fraction * wing
            else:
                g = core

            # Normalise to unit flux and give it the source's own -- normalised after truncation, so
            # a star at the very edge of the frame keeps all of the light that landed on the sensor.
            # What turns a flux into electrons is the detector's business, once, at readout.
            total = g.sum()
            if total <= 0:
                continue
            self.data[ymin:ymax, xmin:xmax] += g / total * ii
