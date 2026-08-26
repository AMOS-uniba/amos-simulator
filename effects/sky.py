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
    Twilight: the Sun below the horizon, its light still scattered into the frame.

    Now in the same currency as everything else -- a surface brightness in magnitudes per square
    arcsecond -- which is what makes it possible to say how bright it should be. It used to be a bare
    `25 * exp(-alt * 5) * exp(-dist**2) * intensity`, and those constants were tuned against a scene
    whose gain was an arbitrary 1e13; read as W/m2 per pixel they came to 4.7e-3 against a dark sky's
    3.8e-12, so switching the term on saturated every pixel of every frame.

    Two numbers now, both configured: the peak surface brightness when the Sun is twelve degrees
    down, and how fast it fades as the Sun sinks further. The defaults put twilight two magnitudes
    above a dark sky at -12 and level with it by -18, so it is plain to see across nautical twilight
    and gone by the end of astronomical twilight.

    **That fade is gentler than the sky's.** Real twilight falls off at closer to a magnitude per
    degree of depression, which over six degrees is a factor of 250 -- overwhelming at one end of the
    band or invisible at the other, whichever end one anchors. 0.4 keeps the whole band worth
    looking at. Steepen `fade` for realism at the cost of a usable range.

    The shape is inherited unchanged: an exponential in altitude times a Gaussian in distance from
    the Sun, so it is brightest at the horizon in the Sun's direction. It has no defence beyond
    looking approximately right, and outside twilight it is extrapolation -- with the Sun up, this
    happily returns a daylit sky, which for a meteor camera is a white frame, correctly.
    """
    #: The depression the configured brightness refers to, in degrees. The end of nautical twilight,
    #: which is where a station's night begins.
    REFERENCE = 12.0

    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        #: Peak surface brightness at REFERENCE degrees of depression, in mag/arcsec2
        self.brightness = kwargs.pop('brightness', 19.5)
        #: Magnitudes fainter per further degree of depression
        self.fade = kwargs.pop('fade', 0.4)
        super().__init__(location, time, **kwargs)

    def peak_brightness(self, depression: float) -> float:
        """ The surface brightness at the brightest point of the sky, in mag/arcsec2. """
        return self.brightness + self.fade * (depression - self.REFERENCE)

    def radiance(self,
                 alt: ArrayLike,
                 az: ArrayLike) -> ArrayLike:
        sun = self.body_altaz('sun')
        peak = brightness.flux_from_surface_brightness(
            self.peak_brightness(-sun.alt.degree), self.pixel_solid_angle)

        distance = self.angular_distance(alt, az, sun)
        # Unity at the horizon in the Sun's direction and falling away from it, both in altitude and
        # in azimuth, which is the one thing the old formula got right
        shape = np.exp(-np.asarray(alt, dtype=float) * 5.0) * np.exp(-distance ** 2)
        return peak * shape
