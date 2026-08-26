"""
The two stages: what the atmosphere takes, and what it gives.

Both of the bugs these guard against were silent. The moon's term was identically zero at every
phase, which looks exactly like a moonless night; the airglow fell towards the horizon instead of
rising, which looks like a slightly different night.
"""
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from effects import airmass, brightness
from effects.sky import Airglow, Emission, Extinction, Moonlight, Sunlight

WHERE = EarthLocation(17.273933 * u.deg, 48.372763 * u.deg, 580.0 * u.m)

#: A night with the moon up: altitude 12 degrees, azimuth 208, phase angle 110.8 degrees.
MOONLIT = Time('2025-10-01T20:20:20')

#: One pixel of a 1600x1200 all-sky frame, in square arcseconds
PIXEL = 1.794e5


def grid(altitudes, azimuth=208.0):
    """ A row of pixels at the given altitudes, all in the same direction. """
    alt = np.radians(np.atleast_2d(np.asarray(altitudes, dtype=float)))
    return alt, np.radians(np.full_like(alt, azimuth))


def one(value) -> float:
    """ The single number out of a one-pixel frame. numpy will not do this with float(). """
    return float(np.asarray(value).ravel()[0])


class TestExtinction:
    def test_it_is_exp_minus_tau(self):
        alt, az = grid([90, 60, 30, 10])
        out = Extinction(WHERE, MOONLIT, extinction=0.145)(np.ones_like(alt), alt, az)
        assert out == pytest.approx(airmass.transmittance(0.145, alt))

    def test_the_zenith_loses_exactly_the_coefficient(self):
        alt, az = grid([90])
        out = Extinction(WHERE, MOONLIT, extinction=0.2)(np.ones_like(alt), alt, az)
        assert one(out) == pytest.approx(10 ** (-0.4 * 0.2), rel=1e-9)

    def test_a_clear_night_takes_less_than_a_hazy_one(self):
        alt, az = grid([30])
        clear = Extinction(WHERE, MOONLIT, extinction=0.145)(np.ones_like(alt), alt, az)
        hazy = Extinction(WHERE, MOONLIT, extinction=0.35)(np.ones_like(alt), alt, az)
        assert one(clear) > one(hazy)


class TestAirglow:
    def radiances(self, altitudes, **kwargs):
        alt, az = grid(altitudes)
        source = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=21.8, **kwargs)
        return np.asarray(source.radiance(alt, az)).ravel()

    def test_the_zenith_is_the_configured_surface_brightness(self):
        """ Bar the one air mass of extinction the light itself has to come through. """
        expected = (brightness.flux_from_surface_brightness(21.8, PIXEL)
                    * airmass.transmittance(0.145, np.pi / 2))
        assert self.radiances([90])[0] == pytest.approx(expected, rel=1e-9)

    def test_it_rises_towards_the_horizon_and_then_falls(self):
        """
        The shape that was backwards before: van Rhijn wins down to about ten degrees, extinction
        below that. What was there fell from 1.00 at the zenith to 0.29 at the horizon.
        """
        values = self.radiances([90, 60, 30, 20, 10, 5, 1])
        peak = int(np.argmax(values))
        assert 3 <= peak <= 5, f"the maximum should be between twenty and five degrees, got {values}"
        assert values[peak] > 2 * values[0]
        assert values[-1] < values[0]

    def test_a_darker_sky_is_dimmer_everywhere(self):
        alt, az = grid([90, 30, 10])
        dark = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=21.8)
        town = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=19.0)
        assert np.all(np.asarray(town.radiance(alt, az)) > np.asarray(dark.radiance(alt, az)))

    def test_a_magnitude_is_a_factor_of_two_and_a_half(self):
        alt, az = grid([45])
        faint = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=21.0)
        two = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=20.0)
        ratio = one(np.asarray(two.radiance(alt, az))) / one(np.asarray(faint.radiance(alt, az)))
        assert ratio == pytest.approx(10 ** 0.4, rel=1e-6)


class TestMoonlight:
    def source(self):
        return Moonlight(WHERE, MOONLIT, pixel_solid_angle=PIXEL, extinction=0.145)

    def test_it_is_not_zero(self):
        """
        The bug this file was written for. `intensity()` converted a phase angle to degrees twice,
        110.8 became 6347, and the whole term underflowed to nothing at every phase.
        """
        alt, az = grid([30, 60])
        assert np.all(np.asarray(self.source().radiance(alt, az)) > 0)

    def test_it_is_brighter_near_the_moon(self):
        alt, az = grid([12, 12], azimuth=208.0)
        near = np.asarray(self.source().radiance(alt, az)).ravel()[0]
        alt, az = grid([12], azimuth=28.0)          # the opposite side of the sky
        far = np.asarray(self.source().radiance(alt, az)).ravel()[0]
        assert near > far

    def test_a_full_moon_is_brighter_than_a_crescent(self):
        full = Moonlight.illuminance(0.0 * u.deg)
        quarter = Moonlight.illuminance(90.0 * u.deg)
        crescent = Moonlight.illuminance(150.0 * u.deg)
        assert full > quarter > crescent > 0

    def test_the_phase_angle_is_converted_once(self):
        """ Degrees in, degrees used: the same angle in radians must not give the same answer. """
        assert (Moonlight.illuminance(110.776 * u.deg)
                == pytest.approx(10 ** (-0.4 * (3.84 + 0.026 * 110.776
                                                + 4e-9 * 110.776 ** 4)), rel=1e-9))
        assert Moonlight.illuminance(np.radians(110.776) * u.rad) == pytest.approx(
            Moonlight.illuminance(110.776 * u.deg), rel=1e-12)

    def test_nothing_when_the_moon_is_down(self):
        alt, az = grid([30, 60])
        # Twelve hours later the moon has set
        down = Moonlight(WHERE, MOONLIT + 0.5 * u.day, pixel_solid_angle=PIXEL)
        moon = down.body_altaz('moon')
        assert moon.alt.degree < 0, "the fixture assumes the moon is below the horizon here"
        assert np.all(np.asarray(down.radiance(alt, az)) == 0)

    def test_it_lands_in_a_believable_surface_brightness(self):
        """
        The check that the nanolambert conversion is the right way up: moonlight thirty degrees from
        a gibbous moon is a few magnitudes brighter than a dark sky, not twelve orders of magnitude.
        """
        alt, az = grid([40], azimuth=208.0)
        flux = one(self.source().radiance(alt, az))
        dark = brightness.flux_from_surface_brightness(21.8, PIXEL)
        assert 0.1 < flux / dark < 100


class TestTheTwoStages:
    def test_an_emission_is_additive_and_stops_at_the_horizon(self):
        alt, az = grid([30, -30])
        source = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=21.8)
        out = np.asarray(source(np.zeros_like(alt), alt, az)).ravel()
        assert out[0] > 0
        assert out[1] == 0

    def test_emission_is_not_attenuated_by_the_extinction_stage(self):
        """
        The ordering bug in one assertion: the sky's own light must not be dimmed as though it had
        arrived from outside the atmosphere. Build the frame in the scene's order and the sky is
        exactly what the emission says.
        """
        alt, az = grid([20])
        glow = Airglow(WHERE, MOONLIT, pixel_solid_angle=PIXEL, brightness=21.8)
        data = Extinction(WHERE, MOONLIT)(np.zeros_like(alt), alt, az)
        assert one(glow(data, alt, az)) == pytest.approx(one(glow.radiance(alt, az)))

    def test_every_emission_is_one(self):
        for source in (Airglow, Moonlight):
            assert issubclass(source, Emission)


class TestTheMoonAsABody:
    """
    The Moon itself, which is a different thing from its glow: the glow is what the air scatters
    sideways, this is the light that came straight down the lens.
    """
    def test_the_phase_law_is_zero_at_full(self):
        assert Moonlight.phase_law(0.0 * u.deg) == 0.0
        assert Moonlight.magnitude(0.0 * u.deg) == pytest.approx(-12.73)

    @pytest.mark.parametrize('phase, expected', [(45, -11.54), (90, -10.13), (110.776, -9.25)])
    def test_and_it_fades_as_the_moon_wanes(self, phase, expected):
        assert Moonlight.magnitude(phase * u.deg) == pytest.approx(expected, abs=0.01)

    def test_the_two_zero_points_differ_by_the_unit_conversion(self):
        """
        The illuminance the scattering model wants and the magnitude of the body are the same phase
        law with different zero points; if one is edited the other has to move with it.
        """
        for phase in (0.0, 60.0, 120.0):
            magnitude = Moonlight.magnitude(phase * u.deg)
            illuminance = Moonlight.illuminance(phase * u.deg)
            assert -2.5 * np.log10(illuminance) - magnitude == pytest.approx(3.84 + 12.73, abs=1e-9)


class TestTwilight:
    """
    The shape and the fall-off are geometry; only the absolute level is a constant. So these check
    the geometry, and the one number is checked by what it implies.
    """
    def source(self, **kwargs):
        return Sunlight(WHERE, MOONLIT, pixel_solid_angle=PIXEL, **kwargs)

    def lit(self, altitude, azimuth_from_sun, depression):
        return float(self.source().lit_fraction(np.array([np.radians(altitude)]),
                                                np.array([np.radians(azimuth_from_sun)]),
                                                np.radians(-depression))[0])

    def test_at_sunset_the_sunward_horizon_is_in_full_sunlight(self):
        assert self.lit(0.5, 0.0, 0.0) == pytest.approx(1.0, abs=0.02)

    def test_the_shadow_climbs_as_the_sun_sinks(self):
        heights = [airmass.shadow_height(np.array([np.radians(1.0)]), np.array([0.0]),
                                         np.radians(-d))[0] for d in (0, 3, 6, 9, 12, 15, 18)]
        assert np.all(np.diff(heights) > 0)
        # And the numbers themselves: 46 km up at the end of nautical twilight, 89 at astronomical
        assert heights[4] == pytest.approx(46.3, rel=0.05)
        assert heights[6] == pytest.approx(88.6, rel=0.05)

    def test_and_stands_higher_away_from_the_sun(self):
        assert self.lit(1.0, 0.0, 12.0) > self.lit(1.0, 90.0, 12.0) > self.lit(1.0, 180.0, 12.0)

    def test_the_observed_rate_falls_out_of_the_geometry(self):
        """
        The check this model exists for. Twilight is measured to fade by about a magnitude for every
        degree the Sun sinks; nothing here was told that, it follows from the shadow rising through an
        atmosphere with an 8 km scale height.
        """
        rate = -2.5 * np.log10(self.lit(1.0, 0.0, 18.0) / self.lit(1.0, 0.0, 12.0)) / 6.0
        assert rate == pytest.approx(1.0, abs=0.15)

    def test_astronomical_twilight_ends_where_it_is_defined_to(self):
        """
        By eighteen degrees the zenith has to be well below a dark sky, which the definition says and
        this has to reproduce rather than assume. The sunward horizon keeps a trace past eighteen,
        which it does in life too, so it is the zenith that is checked.
        """
        s = self.source()
        zenith = (brightness.flux_from_surface_brightness(s.brightness, PIXEL)
                  * (self.lit(89.0, 0.0, 18.0) + s.multiple_scattering * self.lit(1.0, 0.0, 18.0))
                  * 0.125 * 0.75)
        assert zenith < 0.1 * brightness.flux_from_surface_brightness(21.8, PIXEL)

    def test_the_zenith_lands_where_nautical_twilight_is_quoted(self):
        """
        19.5 mag/arcsec2 at twelve degrees of depression, which is about what nautical twilight is
        said to look like. This is the constant's doing, not the geometry's, and it is the number a
        real calibration would move.
        """
        s = self.source()
        flux = (brightness.flux_from_surface_brightness(s.brightness, PIXEL)
                * (self.lit(89.0, 0.0, 12.0) + s.multiple_scattering * self.lit(1.0, 0.0, 12.0))
                * 0.125 * 0.75)
        magnitude = -2.5 * np.log10(flux) - brightness.ZERO_POINT + 2.5 * np.log10(PIXEL)
        assert magnitude == pytest.approx(19.5, abs=0.4)

    def test_the_sunward_horizon_is_the_brightest_part_of_the_sky(self):
        s = Sunlight(WHERE, Time('2025-10-01T17:40:00'), pixel_solid_angle=PIXEL)
        sun = s.body_altaz('sun')
        towards = grid([2.0], azimuth=sun.az.degree)
        away = grid([2.0], azimuth=(sun.az.degree + 180) % 360)
        overhead = grid([88.0], azimuth=sun.az.degree)
        assert one(s.radiance(*towards)) > one(s.radiance(*away))
        assert one(s.radiance(*towards)) > one(s.radiance(*overhead))

    def test_and_the_contrast_grows_as_the_sun_sinks(self):
        """ The glow shrinks towards the sunward horizon rather than fading uniformly. """
        contrasts = [self.lit(1.0, 0.0, d) / self.lit(45.0, 0.0, d) for d in (6, 9, 12)]
        assert contrasts == sorted(contrasts)

    def test_nothing_that_matters_is_lit_in_the_middle_of_the_night(self):
        """
        Not `inf`: a ray does leave the shadow eventually, a couple of thousand kilometres out, and
        the honest statement is that there is no air there. The exponential says so exactly.
        """
        height = airmass.shadow_height(np.array([0.5]), np.array([0.0]), np.radians(-60.0))[0]
        assert height > 500.0
        # 1e-124 of the column, which is zero in every sense but the arithmetic's
        assert self.lit(30.0, 0.0, 60.0) < 1e-30

    def test_and_straight_up_at_midnight_the_ray_never_leaves_it(self):
        assert np.isinf(airmass.shadow_height(np.array([np.radians(89.0)]), np.array([0.0]),
                                              np.radians(-60.0))[0])


class TestThePhaseAngle:
    """
    The Sun-Moon-Earth angle, and the mistake it is easy to make: astropy's `separation` gives the
    *elongation*, and the phase angle is its supplement. Getting that wrong turns every phase into
    its opposite, which is what this model did from the day it was written.
    """
    def test_it_is_the_supplement_of_the_elongation(self):
        from astropy.coordinates import get_body
        for when in ('2025-09-21T20:00:00', '2025-10-07T20:00:00', '2025-10-16T02:00:00'):
            t = Time(when)
            elongation = get_body('moon', t, WHERE).separation(get_body('sun', t, WHERE))
            phase = Moonlight.phase_angle(WHERE, t)
            assert (elongation + phase).to(u.deg).value == pytest.approx(180.0, abs=1e-6)

    def test_the_full_moon_of_october_2025(self):
        """ It was on the 7th. Ninety-nine percent lit, and within half a magnitude of -12.7. """
        phase = Moonlight.phase_angle(WHERE, Time('2025-10-07T20:00:00'))
        lit = (1 + np.cos(phase.to(u.rad).value)) / 2
        assert lit > 0.98
        assert Moonlight.magnitude(phase) == pytest.approx(-12.5, abs=0.5)

    def test_and_the_new_moon_that_started_that_month(self):
        """
        2025-09-21, and the point of the test: read as a phase angle rather than an elongation, that
        night's 1.8 degrees made the model draw a *full* Moon at V = -12.68.
        """
        phase = Moonlight.phase_angle(WHERE, Time('2025-09-21T20:00:00'))
        lit = (1 + np.cos(phase.to(u.rad).value)) / 2
        assert lit < 0.01
        assert Moonlight.magnitude(phase) > -5.0

    def test_the_moon_of_the_configured_epoch_is_a_waxing_gibbous(self):
        """ 2025-10-01, which every other test in this repository is set at: 68% lit, not 32%. """
        phase = Moonlight.phase_angle(WHERE, MOONLIT)
        lit = (1 + np.cos(phase.to(u.rad).value)) / 2
        assert lit == pytest.approx(0.676, abs=0.01)
        assert Moonlight.magnitude(phase) == pytest.approx(-10.83, abs=0.05)
