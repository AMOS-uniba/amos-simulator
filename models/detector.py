"""
From a flux on the sky to the number in a pixel, one stage at a time.

Everything here is in electrons, which is the point. What this replaces was three calls with
hardcoded arguments -- `render_as_poisson()`, `add_intensifier_noise(100, 10, radius=70)`,
`add_thermal_noise(rate=40)` -- and a `flux_to_electrons(lambda x: x * 10)` where the conversion
should have been, so no number in the chain could be compared with anything. One of them was worse
than arbitrary: `add_thermal_noise` computed `20 * Poisson(40)`, and multiplying a Poisson variate
by twenty gives the right mean with four hundred times the variance. To model eight hundred
electrons one draws `Poisson(800)`.

The stages, in the order the photons meet them:

  1. flux to photoelectrons -- aperture, quantum efficiency, exposure, and the energy of a photon
  2. shot noise -- Poisson, because photons arrive independently
  3. dark current -- Poisson at a rate times the exposure
  4. the intensifier -- a gain with variance in excess of Poisson, plus its own glow in the middle
     of the field
  5. read noise -- Gaussian, and the one term that does not care how long the shutter was open
  6. bias, then the full well, then quantisation to the output depth

The full well is what makes bright stars bloom, and it earns `Scene.intensity_to_sigma` its
retirement: a flat-topped core is a clipped core, not a wider Gaussian.
"""
import logging

import numpy as np
from numpy.typing import ArrayLike

#: Planck constant times the speed of light, in J*nm, so that a wavelength in nanometres gives the
#: energy of one photon in joules.
HC = 1.98644586e-16

#: The same camera `config/renderers/default.yaml` describes, so that a Scene built by hand -- a
#: test, a notebook -- behaves like the renderer rather than like an earlier draft of it. The one
#: difference is `glow`, which defaults to nothing: the intensifier's haze is a thing one asks for.
DEFAULTS = dict(
    aperture=4.3, qe=0.20, wavelength=550.0,
    exposure=0.04, dark=5.0, read_noise=200.0, bias=500.0, full_well=128000.0, bits=8,
    gain=10000.0, excess_noise=2.0, glow=0.0, glow_radius=70.0,
    # Only read when `gain: auto`: what the loop aims for, measured at a high percentile of the
    # frame so that a bright rim sets it and a handful of saturated stars do not, and the range a
    # tube can actually be driven over.
    agc_target=60.0, agc_percentile=90.0, agc_min=50.0, agc_max=20000.0,
)

log = logging.getLogger('root')


class Detector:
    """
    One camera's readout chain, built from the `optics` and `detector` blocks of a renderer config.
    """
    def __init__(self, optics: dict = None, detector: dict = None):
        settings = dict(DEFAULTS)
        settings.update({k: v for k, v in (optics or {}).items() if k in DEFAULTS})
        settings.update({k: v for k, v in (detector or {}).items() if k in DEFAULTS})
        settings.update({k: v for k, v in ((detector or {}).get('intensifier') or {}).items()
                         if k in DEFAULTS})
        self.__dict__.update(settings)

    @property
    def area(self) -> float:
        """ Collecting area of the entrance pupil, in square metres, from a diameter in mm. """
        return np.pi * (self.aperture * 1e-3 / 2.0) ** 2

    @property
    def photon_energy(self) -> float:
        """ Energy of one photon at the effective wavelength, in joules. """
        return HC / self.wavelength

    @property
    def saturation(self) -> int:
        """ The largest number the output can hold. """
        return 2 ** int(self.bits) - 1

    @property
    def automatic(self) -> bool:
        """ Whether the intensifier turns itself down when the sky is bright. """
        return isinstance(self.gain, str) and self.gain.lower() == 'auto'

    def choose_gain(self, electrons: ArrayLike) -> float:
        """
        What an automatic gain control would settle on for a frame carrying these electrons.

        A real intensified camera does not run at one gain all night: the tube is driven by a loop
        that watches the output and turns the voltage down as the sky brightens, which is both how a
        station survives twilight and why it keeps working through it. Without that there is no gain
        that suits both a dark sky and nautical twilight -- two hundred to one across an 8-bit
        output -- so a fixed number has to give up one end or the other.

        Set from a high percentile rather than the mean, so that a bright rim decides it and a few
        saturated stars do not, and clamped to the range a tube actually has.

        **A frame taken this way is not photometric**: two frames an hour apart were taken at
        different gains and their counts are not comparable. That is a real property of the hardware
        and the reason `default.yaml`, where the brightness of a meteor is the measurement, pins the
        gain instead.
        """
        if not self.automatic:
            return float(self.gain)

        level = float(np.percentile(np.asarray(electrons, dtype=float), self.agc_percentile))
        if level <= 0.0:
            return float(self.agc_max)
        wanted = self.agc_target * self.full_well / ((self.saturation + 1) * level)
        return float(np.clip(wanted, self.agc_min, self.agc_max))

    @property
    def smallest_flux(self) -> float:
        """
        The flux in one pixel that the output can just about tell from nothing: one count.

        What a profile has to fall below before there is no point drawing it any further, which is
        how far a saturated source spreads.
        """
        # At the most sensitive the tube can be, when it is the tube that decides: the radius this
        # feeds is a bound on how far to draw a source, and a bound wants the generous case.
        gain = self.agc_max if self.automatic else max(self.gain, 1e-30)
        per_count = self.full_well / (self.saturation + 1) / gain
        return per_count * self.photon_energy / (self.area * self.exposure * self.qe)

    def photoelectrons(self, flux: ArrayLike) -> ArrayLike:
        """
        Photoelectrons collected from a flux in W/m2 over one exposure.

        This is the whole of what `Scene.gain = 1e13` used to assert. With the defaults a zeroth
        magnitude star gives 3561 photoelectrons in 40 ms, and since a pixel holds about thirteen of
        them before the intensifier's gain fills the well, it saturates -- as it should, a bright
        star on a real AMOS frame being a saturated blob.
        """
        return (np.asarray(flux, dtype=float) * self.area * self.exposure
                * self.qe / self.photon_energy)

    def glow_profile(self, xs: ArrayLike, ys: ArrayLike) -> ArrayLike:
        """
        The intensifier's own haze, brightest in the middle of the field.

        A Gaussian in radius, as before, but a rate in electrons per second rather than a multiplier
        on a drawn variate, so it goes through the same Poisson draw as everything else.
        """
        xc, ys_shape = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        cx, cy = (xc.shape[1] - 1) / 2.0, (xc.shape[0] - 1) / 2.0
        return np.exp(-((xc - cx) ** 2 + (ys_shape - cy) ** 2) / (2.0 * self.glow_radius ** 2))

    def readout(self, flux: ArrayLike, xs: ArrayLike, ys: ArrayLike) -> ArrayLike:
        """
        A frame's worth of flux, in W/m2 per pixel, through the whole chain to output counts.
        """
        signal = self.photoelectrons(flux)
        dark = self.dark * self.exposure
        haze = self.glow * self.exposure * self.glow_profile(xs, ys)

        # One Poisson draw for everything that arrives as independent events: the star light, the
        # thermal electrons of the sensor and the intensifier's own thermionic haze.
        electrons = np.random.poisson(signal + dark + haze).astype(float)

        # The intensifier multiplies, and its variance exceeds Poisson's by the excess noise factor.
        # Adding the missing variance as a Gaussian is the standard cheap stand-in for the gamma
        # distribution an MCP actually produces, and is indistinguishable once a frame is quantised.
        gain = self.choose_gain(electrons)
        if gain != 1.0 or self.excess_noise > 1.0:
            extra = max(self.excess_noise - 1.0, 0.0) * electrons
            electrons = gain * (electrons + np.random.normal(0.0, np.sqrt(extra)))
        if self.automatic:
            log.info(f"Gain {gain:.0f}")

        electrons += np.random.normal(0.0, self.read_noise, size=electrons.shape)
        electrons += self.bias

        # Where a pixel stops counting. Everything above this is the same number, which is what
        # blooming looks like once it has been written down.
        electrons = np.clip(electrons, 0.0, self.full_well)

        counts = np.floor(electrons / self.full_well * (self.saturation + 1))
        return np.clip(counts, 0, self.saturation)
