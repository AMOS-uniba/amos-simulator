"""
Turning what the literature quotes into what the scene adds up.

`Scene.data` holds a flux per pixel in W/m², so anything arriving in another currency has to be
converted before it can be added. Three currencies show up:

  * **magnitudes** -- a star, straight out of a catalogue
  * **magnitudes per square arcsecond** -- how sky brightness is always quoted, and the only unit in
    which "a dark rural sky" means something
  * **nanolamberts** -- what Krisciunas & Schaefer's moonlight model returns, because photometry in
    1991 quoted the night sky in photometric brightness units

The middle two need the pixel's own patch of sky, which is why they take a solid angle: a surface
brightness says nothing about a frame until you know how much sky one pixel sees.

This exists because the constants of the moonlight model were being added to a flux array as though
they were W/m². While `Scene.gain` was an arbitrary 1e13 nobody could tell; once a flux became a
number of photoelectrons, the twilight term alone was contributing 4.7e-3 W/m² per pixel against a
sky of 3.8e-12, and every frame came out saturated.
"""
import numpy as np
from numpy.typing import ArrayLike

#: The zero point that makes a magnitude a flux in W/m²: V = 0 is 1.1e-8 W/m² in a broad visual
#: band. Inherited from `Scene.vmag_to_intensity`, which is where the ecosystem's magnitudes have
#: always been converted, and kept identical to it on purpose.
ZERO_POINT = 19.89


def flux_from_magnitude(magnitude: ArrayLike) -> ArrayLike:
    """ A visual magnitude as a flux in W/m². """
    return 10 ** (-0.4 * (np.asarray(magnitude, dtype=float) + ZERO_POINT))


def flux_from_surface_brightness(magnitude: ArrayLike, pixel_solid_angle: float) -> ArrayLike:
    """
    A surface brightness in magnitudes per square arcsecond, as the flux one pixel receives.

    21.8 is a dark rural zenith, 20.5 a village, 18 a city centre. With a pixel 424 arcseconds across
    -- which is what an all-sky lens on a 1600x1200 sensor gives -- that first one comes to
    3.8e-12 W/m² in a pixel.
    """
    return flux_from_magnitude(np.asarray(magnitude, dtype=float)
                               - 2.5 * np.log10(pixel_solid_angle))


def magnitude_from_nanolamberts(brightness: ArrayLike) -> ArrayLike:
    """
    Nanolamberts to magnitudes per square arcsecond, which is Krisciunas & Schaefer's own inversion.

    Their moonlight model is built in nanolamberts throughout, so this is the one place its output
    turns into something this renderer can use. 34.08 nL is roughly V = 21.6 per square arcsecond, a
    dark sky, which is a useful sanity check on the constant.
    """
    return (20.7233 - np.log(np.asarray(brightness, dtype=float) / 34.08)) / 0.92104
