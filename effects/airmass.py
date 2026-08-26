"""
How far light travels through the atmosphere on its way to a camera looking at a given altitude.

Two path lengths live here, and they are not interchangeable. Which one an effect wants follows from
where the light comes from:

  * `kasten_young` is the air mass: the whole column, normalised to one at the zenith. It is what
    attenuates anything arriving from outside the atmosphere -- a star, a planet, a meteor -- and
    what attenuates light on its way out of the atmosphere too.
  * `van_rhijn` is the slant path through a thin shell at a given height, which is the geometry of
    airglow. It grows to about six at the horizon where the air mass grows to thirty-eight, because a
    ray leaving the horizon crosses the emitting layer obliquely but only once.

There used to be two air masses in `effects/sky.py`, disagreeing: one rational fit that tracked
Kasten-Young well and returned a sentinel 1000 below the horizon, and one written as
`(1 - 0.96 cos(alt))**-0.5`, which wants `cos**2` and so was wrong everywhere but the zenith --
2.44 against 1.99 at thirty degrees, and saturating at 5.0 where the truth is 37.9. The wrong one was
the one `Airglow` and `Moonlight` inherited. `config/renderers/*.yaml` named the formula that should
have been there all along and nothing read the key.
"""
import numpy as np
from numpy.typing import ArrayLike

#: Earth's mean radius, in kilometres. Only the ratio to the layer height matters.
EARTH_RADIUS = 6371.0

#: Where the airglow comes from: the mesopause, near 90 km, where OH and O2 emit at night. A number
#: worth having in one place, since the same height sets the horizon enhancement.
AIRGLOW_HEIGHT = 90.0

#: Magnitudes lost per air mass at the zenith, in V, on a clear night at a few hundred metres. The
#: default only; every effect takes it from the configuration.
EXTINCTION = 0.145


def _kasten_young(zenith_deg: ArrayLike) -> ArrayLike:
    """ Kasten & Young (1989) as published, with the zenith angle in degrees. """
    return 1.0 / (np.cos(np.radians(zenith_deg)) + 0.50572 * (96.07995 - zenith_deg) ** -1.6364)


#: What the formula answers looking straight up: 0.99971, not 1. Its second term does not vanish at
#: the zenith, so it is normalised below -- a correction of three parts in ten thousand, far inside
#: the formula's own accuracy, and worth making because it is what lets an extinction coefficient
#: mean what an observer means by it: magnitudes lost looking straight up, exactly.
ZENITH = _kasten_young(0.0)


def kasten_young(alt: ArrayLike) -> ArrayLike:
    """
    Air mass at altitude `alt` (radians), after Kasten & Young (1989), normalised at the zenith.

    Exact enough to be the only one: 1.000 at the zenith by construction, 1.154 at sixty degrees,
    5.588 at ten, 37.93 at the horizon. Below the horizon the zenith angle is clamped, both because
    the formula's `(96.07995 - z)**-1.6364` turns into a negative base a few degrees down and because
    nothing is drawn there anyway -- an effect that reaches below the horizon has already asked the
    wrong question.
    """
    zenith = np.clip(90.0 - np.degrees(alt), 0.0, 90.0)
    return _kasten_young(zenith) / ZENITH


def van_rhijn(alt: ArrayLike,
              height: float = AIRGLOW_HEIGHT,
              radius: float = EARTH_RADIUS) -> ArrayLike:
    """
    Slant path through a shell at `height` km, relative to looking straight up at it.

    The airglow's own geometry: a ray towards the horizon crosses the layer at a shallow angle and
    so sees more of it, but the layer is thin and the enhancement saturates -- 1.92 at thirty
    degrees, 4.19 at ten, 6.01 at the horizon for 90 km. Using the air mass instead would claim
    thirty-eight, which is the mistake this function exists to avoid.
    """
    zenith = np.clip(np.pi / 2 - np.asarray(alt, dtype=float), 0.0, np.pi / 2)
    ratio = radius / (radius + height)
    return 1.0 / np.sqrt(1.0 - (ratio * np.sin(zenith)) ** 2)


def optical_depth(extinction: float, airmass: ArrayLike) -> ArrayLike:
    """
    Optical depth from an extinction coefficient in magnitudes per air mass.

    The only conversion in the file, and the reason `exp(-tau)` and `10**(-0.4 k X)` are the same
    statement: a magnitude is 0.4 * ln(10) of an e-folding.
    """
    return 0.4 * np.log(10.0) * extinction * np.asarray(airmass, dtype=float)


def transmittance(extinction: float, alt: ArrayLike) -> ArrayLike:
    """
    The fraction of light from outside the atmosphere that survives the trip to altitude `alt`.
    """
    return np.exp(-optical_depth(extinction, kasten_young(alt)))


def slab_radiance(source: ArrayLike, tau: ArrayLike) -> ArrayLike:
    """
    What a uniformly emitting, self-absorbing column of optical depth `tau` shines at the observer.

    `source * (1 - exp(-tau))`, which is the whole of radiative transfer for a medium that does not
    scatter and has one source function: emission accumulates along the path and is reabsorbed by the
    same medium, so it saturates at the source function instead of growing without bound.

    The linear limit `source * tau` is what one writes first and it is a good approximation while the
    column is thin -- 7% high at the zenith, 8% at sixty degrees, 14% at thirty. It is 42% high at
    ten degrees and a factor of 5.1 high at the horizon, which is exactly where a wide-field camera
    spends most of its pixels. This costs one exponential more.
    """
    return np.asarray(source, dtype=float) * -np.expm1(-np.asarray(tau, dtype=float))
