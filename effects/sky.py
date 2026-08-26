"""
What the atmosphere does to a frame: it takes light away, and it adds light of its own.

Those are the two stages, and the split is the point of this module. Everything arriving from
outside -- stars, planets, the meteor -- is dimmed by `Extinction`, once, over the whole column.
Everything generated inside is an `Emission` and is added afterwards: the airglow, the moon's light
scattered towards the camera, the last of the twilight, and light pollution when it is written.

It used to be one list applied in order, `[sun, moon, extinction, airglow]`, which made the ordering
load-bearing in a way nobody had chosen: the sun and moon terms are scattered light, generated inside
the atmosphere, and being ahead of `extinction` in the list they were dimmed as though they had come
from outside it, while the airglow, being behind it, was not dimmed at all.

Path lengths are `effects/airmass.py`, which is also where the note about the two of them lives.
"""
import numpy as np
from abc import abstractmethod
from typing import Optional

from demeteor.metrics import spherical
from numpy.typing import ArrayLike

from astropy.coordinates import EarthLocation, get_body, AltAz
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

from effects import airmass, brightness


class SkyEffect:
    """
    Something the atmosphere does to the light reaching each pixel, in terms of that pixel's
    altitude and azimuth. Either takes light away (`Extinction`) or adds it (`Emission`).
    """
    def __init__(self,
                 location: EarthLocation,
                 time: Optional[Time] = None,
                 **kwargs):
        self.location = location
        self.time = time if time is not None else Time.now()
        self.altaz = AltAz(obstime=self.time, location=self.location)
        self.extinction = kwargs.pop('extinction', airmass.EXTINCTION)
        # How much sky one pixel sees, in square arcseconds. Anything quoting a surface brightness
        # needs it; the scene measures it off the projection and hands it down.
        self.pixel_solid_angle = kwargs.pop('pixel_solid_angle', 1.794e5)

    @abstractmethod
    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        """
        Transform the value at `data` by whatever happens at (`alt`, `az`).
        """

    def body_altaz(self, body: str):
        """ Where the sun or the moon is, from here, now. """
        return get_body(body, self.time, self.location).transform_to(self.altaz)

    @staticmethod
    def angular_distance(alt: ArrayLike, az: ArrayLike, target) -> ArrayLike:
        """
        Angle from every pixel to a body. The pixel arrays are (y, x), so the coordinate pair goes on
        a new last axis -- which is what `spherical` expects and what the old `axis=2` happened to
        mean for a two-dimensional frame and nothing else.
        """
        pixels = np.stack((alt, az), axis=-1)
        return spherical(pixels, np.array([target.alt.radian, target.az.radian]))


class Extinction(SkyEffect):
    """
    Attenuation of everything that came from outside the atmosphere: `exp(-tau)` over the air mass.

    Multiplicative, and applied once, to the sum of the sources rather than to each of them -- there
    is only one column and every photon in a pixel crossed the same amount of it.
    """
    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        return data * airmass.transmittance(self.extinction, alt)


class Emission(SkyEffect):
    """
    Light the atmosphere makes or redirects, added to the frame.

    Purely additive, and zero below the horizon -- what is down there is the ground, and the ground is
    not this module's business. Light pollution will be one of these.
    """
    def __call__(self,
                 data: ArrayLike,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        return data + np.where(alt > 0, self.radiance(alt, az), 0.0)

    @abstractmethod
    def radiance(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        """
        What this source shines at the camera from (`alt`, `az`), in the scene's flux units.
        """


class Airglow(Emission):
    """
    The night sky's own light: OH and O2 emitting in a layer around 90 km.

    A layer and not a column, so the horizon enhancement is van Rhijn's -- about six at the horizon,
    not the air mass's thirty-eight -- and what leaves the layer is then dimmed by the whole
    atmosphere beneath it. The two together rise from the zenith to a maximum near ten degrees and
    fall below it, which is what one sees.

    The version this replaces multiplied by an air mass and dimmed by `10**(-0.4 k (X - 1))`, giving
    an airglow that fell from 1.00 at the zenith to 0.29 at the horizon: brightest overhead, which is
    backwards.
    """
    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        #: Zenith surface brightness, in magnitudes per square arcsecond
        self.brightness = kwargs.pop('brightness', 21.8)
        self.height = kwargs.pop('height', airmass.AIRGLOW_HEIGHT)
        super().__init__(location, time, **kwargs)

    def radiance(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        zenith = brightness.flux_from_surface_brightness(self.brightness, self.pixel_solid_angle)
        layer = airmass.van_rhijn(alt, height=self.height)
        return zenith * layer * airmass.transmittance(self.extinction, alt)


class Moonlight(Emission):
    """
    Moonlight scattered towards the camera by the air between it and the sky.

    Krisciunas & Schaefer (1991), in three parts: how much light the moon sends at this phase, how
    much of it a column at this angular distance scatters towards us, and what the atmosphere takes on
    the way in and on the way to the camera. The scattering medium is the whole lower atmosphere, so
    the emission saturates as `1 - exp(-tau)` rather than growing with the air mass.

    This used to return exactly zero, at every phase and every altitude. `intensity()` called
    `np.degrees` on a phase angle the caller had already converted, so 110.8 degrees arrived as 6347,
    the `4e-9 * alpha**4` term reached 6.5e6, and `10**(-0.4 * that)` underflowed. The brightest thing
    in the model was switched off by a unit conversion. The angle is now converted once, here, from an
    astropy quantity that carries its own unit.
    """
    @staticmethod
    def scattering(distance: ArrayLike) -> ArrayLike:
        """
        The scattering function: Rayleigh at large angles, plus the aureole close to the moon.
        """
        return 10 ** 5.36 * (1.06 + np.cos(distance) ** 2) + 10 ** (6.15 - np.degrees(distance) / 40)

    #: The full moon, in V. The phase law below is written once and used with two zero points: this
    #: one for the magnitude of the body itself, and Krisciunas & Schaefer's 3.84 for the illuminance
    #: their scattering model is built in. The 16.57 magnitudes between them are the conversion from
    #: their photometric units to a flux, and having both here is what keeps them consistent.
    FULL = -12.73

    @staticmethod
    def phase_law(phase: u.Quantity) -> float:
        """
        How many magnitudes fainter than full the moon is at phase angle `phase`.

        Krisciunas & Schaefer (1991): 0.026 per degree, plus a quartic that only matters near new.
        Zero at full, 2.6 at quarter, 5.4 at a thin crescent.
        """
        alpha = float(np.abs(phase.to(u.deg).value))
        return 0.026 * alpha + 4e-9 * alpha ** 4

    @classmethod
    def illuminance(cls, phase: u.Quantity) -> float:
        """
        The moon's own brightness at phase angle `phase`, in the units the scattering model wants.
        """
        return 10 ** (-0.4 * (3.84 + cls.phase_law(phase)))

    @classmethod
    def magnitude(cls, phase: u.Quantity) -> float:
        """
        The moon's apparent visual magnitude at phase angle `phase`: -12.73 at full, -10.1 at
        quarter, -9.3 for the gibbous moon of 2025-10-01.
        """
        return cls.FULL + cls.phase_law(phase)

    def radiance(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        moon = self.body_altaz('moon')
        if moon.alt.radian <= 0:
            return np.zeros_like(np.asarray(alt, dtype=float))

        sun = get_body('sun', self.time, self.location)
        phase = get_body('moon', self.time, self.location).separation(sun)

        # What reaches the air above us, the moon's own air mass having taken its share on the way in
        incident = self.illuminance(phase) * airmass.transmittance(self.extinction, moon.alt.radian)

        distance = self.angular_distance(alt, az, moon)
        tau = airmass.optical_depth(self.extinction, airmass.kasten_young(alt))

        # Krisciunas & Schaefer work in nanolamberts from end to end, and this is where that stops:
        # a surface brightness, then the flux a pixel of this camera receives from it. Adding their
        # constants straight into a flux array was what saturated every frame once the gain became a
        # real number of electrons rather than 1e13.
        nanolamberts = airmass.slab_radiance(self.scattering(distance) * incident, tau)
        magnitudes = brightness.magnitude_from_nanolamberts(np.maximum(nanolamberts, 1e-6))
        return brightness.flux_from_surface_brightness(magnitudes, self.pixel_solid_angle)


class Sunlight(Emission):
    """
    Twilight: the Sun below the horizon, its light scattered into the frame by the air that is still
    in sunlight.

    The structure and the fall-off are geometry, not fitted constants. A line of sight leaves the
    Earth's shadow at some height -- `airmass.shadow_height` -- and only the air above that point can
    scatter anything towards the camera. The density above a height falls off with an 8 km scale
    height, so the brightness goes as `exp(-H_shadow / 8 km)`, and as the Sun sinks the shadow climbs
    and the sky darkens.

    **That reproduces the observed rate without being told it.** On the sunward horizon the shadow
    rises from 46 km at twelve degrees of depression to 89 km at eighteen, which is a factor of 199
    in density above it: 5.75 magnitudes over six degrees, or 0.96 per degree, which is what twilight
    is measured to do. It also gets the shape right: the shadow is lowest towards the Sun and near
    the horizon, so that is where the glow is, and it sinks as the Sun does.

    Single scattering alone would make the zenith far darker than it is -- at twelve degrees down the
    shadow is 154 km up at the zenith, where there is nothing left to scatter -- because the zenith at
    that hour is lit by light that has bounced more than once. `multiple_scattering` is that: a
    diffuse floor, taken as a fraction of the sunward horizon's direct term, which is the crudest
    honest stand-in for a radiative transfer calculation.

    Two constants, then, and they are the fittable part of this: `brightness` sets the level and
    `multiple_scattering` the horizon-to-zenith contrast. **The rate is not among them** -- it comes
    out of the shadow geometry at 0.97 magnitudes per degree whatever these two are set to, which is
    what makes the model worth more than the three tuned numbers it replaced.

    `brightness` is the surface brightness of the sunward horizon at sunset, in magnitudes per square
    arcsecond, and it sets the level of everything. 9.7 puts the zenith at 19.5 mag/arcsec2 with the
    Sun twelve degrees down and 25.4 -- three and a half magnitudes below a dark sky, so invisible --
    at eighteen. The sunward horizon keeps a trace past eighteen, which it does in life too.

    **Neither constant is measured, and they should be.** They come from what the sky is generally
    said to look like at nautical twilight, not from a photometric table, and the horizon-to-zenith
    contrast this pair gives, 4.2 magnitudes, is at the steep end of plausible. Patat et al. (2006)
    measured UBVRI twilight at Paranal against solar depression and is what to fit against; a real
    frame from the station in question would do as well. The two numbers are in config/renderers,
    which is where a fit would land.
    """
    #: Where the shadow-height grid is evaluated before being interpolated onto the pixels. The
    #: twilight field is smooth in altitude and in azimuth from the Sun, so this is plenty, and it
    #: turns 1.9 million rays into two thousand.
    GRID = 48

    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        #: Surface brightness of the sunward horizon at sunset, mag/arcsec2
        self.brightness = kwargs.pop('brightness', 9.7)
        #: Fraction of it that reaches the rest of the sky by scattering more than once
        self.multiple_scattering = kwargs.pop('multiple_scattering', 0.4)
        super().__init__(location, time, **kwargs)

    def lit_fraction(self, alt: ArrayLike, delta_az: ArrayLike, sun_alt: float) -> ArrayLike:
        """
        How much scattering air a line of sight has above the shadow, relative to the whole column.
        """
        height = airmass.shadow_height(alt, delta_az, sun_alt)
        return np.exp(-height / airmass.SCALE_HEIGHT)

    def radiance(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        sun = self.body_altaz('sun')
        delta = np.abs(np.asarray(az, dtype=float) - sun.az.radian) % (2.0 * np.pi)
        delta = np.minimum(delta, 2.0 * np.pi - delta)

        # On a coarse grid, then interpolated: the field is smooth and the march is per-ray
        axis_alt = np.linspace(0.0, np.pi / 2, self.GRID)
        axis_az = np.linspace(0.0, np.pi, self.GRID)
        mesh_alt, mesh_az = np.meshgrid(axis_alt, axis_az, indexing='ij')
        grid = self.lit_fraction(mesh_alt, mesh_az, sun.alt.radian)
        direct = RegularGridInterpolator((axis_alt, axis_az), grid, bounds_error=False,
                                         fill_value=None)(
            np.stack([np.clip(alt, 0.0, np.pi / 2), delta], axis=-1))

        # What has bounced more than once, taken as a fraction of the brightest part of the sky
        horizon = float(self.lit_fraction(np.array([0.0]), np.array([0.0]), sun.alt.radian)[0])
        scattered = direct + self.multiple_scattering * horizon

        # Rayleigh, normalised to one at ninety degrees: the air scatters forwards and backwards
        # about twice as well as sideways
        angle = self.angular_distance(alt, az, sun)
        rayleigh = 0.75 * (1.0 + np.cos(angle) ** 2)

        # How much air is on this line of sight at all, saturating as it thickens -- the same slab
        # the moonlight uses, and for the same reason: this is light made along the ray, not light
        # arriving through it. Attenuating it by the full column instead, as an external source, put
        # the zenith *brighter* than the sunward horizon: the horizon's nineteen air masses took
        # eleven times as much away as its longer lit path had gained.
        tau = airmass.optical_depth(self.extinction, airmass.kasten_young(alt))
        column = airmass.slab_radiance(1.0, tau)

        peak = brightness.flux_from_surface_brightness(self.brightness, self.pixel_solid_angle)
        return peak * scattered * column * rayleigh
