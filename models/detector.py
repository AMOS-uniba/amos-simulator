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

DEFAULTS = dict(
    aperture=3.6, qe=0.45, wavelength=550.0,
    exposure=0.04, dark=1200.0, read_noise=8.0, bias=200.0, full_well=30000.0, bits=8,
    gain=1.0, excess_noise=2.0, glow=0.0, glow_radius=70.0,
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

    def photoelectrons(self, flux: ArrayLike) -> ArrayLike:
        """
        Photoelectrons collected from a flux in W/m2 over one exposure.

        This is the whole of what `Scene.gain = 1e13` used to assert. With the defaults a zeroth
        magnitude star gives some 7e5 electrons in 40 ms, which is well past the full well -- as it
        should be, since a bright star on a real AMOS frame is a saturated blob.
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
        if self.gain != 1.0 or self.excess_noise > 1.0:
            extra = max(self.excess_noise - 1.0, 0.0) * electrons
            electrons = self.gain * (electrons + np.random.normal(0.0, np.sqrt(extra)))

        electrons += np.random.normal(0.0, self.read_noise, size=electrons.shape)
        electrons += self.bias

        # Where a pixel stops counting. Everything above this is the same number, which is what
        # blooming looks like once it has been written down.
        electrons = np.clip(electrons, 0.0, self.full_well)

        counts = np.floor(electrons / self.full_well * (self.saturation + 1))
        return np.clip(counts, 0, self.saturation)
