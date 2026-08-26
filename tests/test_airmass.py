"""
Path lengths through the atmosphere.

There used to be two air masses here, disagreeing, and the wrong one was the one two of the three
effects inherited. These are the numbers that would have said so.
"""
import numpy as np
import pytest

from effects import airmass


class TestKastenYoung:
    #: Published air mass at these altitudes. Kasten & Young (1989), which is also the formula
    #: config/renderers/*.yaml has always named.
    TABLE = {90: 1.000, 60: 1.154, 30: 1.994, 20: 2.904, 10: 5.586, 5: 10.32, 0: 37.92}

    @pytest.mark.parametrize('altitude, expected', sorted(TABLE.items()))
    def test_against_the_published_value(self, altitude, expected):
        assert airmass.kasten_young(np.radians(altitude)) == pytest.approx(expected, rel=2e-3)

    def test_the_zenith_is_one_by_definition(self):
        assert airmass.kasten_young(np.pi / 2) == pytest.approx(1.0, abs=1e-6)

    def test_it_grows_all_the_way_down(self):
        altitudes = np.radians(np.array([90, 60, 30, 10, 5, 1, 0]))
        values = airmass.kasten_young(altitudes)
        assert np.all(np.diff(values) > 0)

    def test_below_the_horizon_it_clamps_rather_than_exploding(self):
        """
        `(96.07995 - z)**-1.6364` turns into a negative base a few degrees below the horizon and the
        old version answered a sentinel 1000 there. Nothing is drawn below the horizon, so the value
        only has to be finite and not a special case anybody has to remember.
        """
        below = airmass.kasten_young(np.radians(np.array([-1.0, -20.0, -90.0])))
        assert np.all(np.isfinite(below))
        assert np.all(below == pytest.approx(airmass.kasten_young(0.0)))


class TestVanRhijn:
    def test_the_zenith_is_one(self):
        assert airmass.van_rhijn(np.pi / 2) == pytest.approx(1.0, abs=1e-9)

    def test_the_horizon_is_six_and_not_thirty_eight(self):
        """
        The whole reason this function exists beside the air mass: a ray towards the horizon crosses
        the emitting layer obliquely, but only once, so the enhancement saturates.
        """
        assert airmass.van_rhijn(0.0) == pytest.approx(6.01, rel=0.01)
        assert airmass.van_rhijn(0.0) < airmass.kasten_young(0.0) / 5

    @pytest.mark.parametrize('altitude, expected', [(60, 1.149), (30, 1.922), (10, 4.189)])
    def test_between_them(self, altitude, expected):
        assert airmass.van_rhijn(np.radians(altitude)) == pytest.approx(expected, rel=2e-3)

    def test_a_higher_layer_is_seen_less_obliquely(self):
        assert airmass.van_rhijn(0.0, height=300.0) < airmass.van_rhijn(0.0, height=90.0)


class TestOpticalDepth:
    def test_a_magnitude_per_airmass_is_a_factor_of_ten_over_2_5(self):
        """ The conversion this file exists to hold in one place. """
        tau = airmass.optical_depth(1.0, 1.0)
        assert np.exp(-tau) == pytest.approx(10 ** -0.4, rel=1e-12)

    def test_transmittance_at_the_zenith_is_the_extinction_coefficient(self):
        assert airmass.transmittance(0.145, np.pi / 2) == pytest.approx(10 ** (-0.4 * 0.145), rel=1e-9)

    def test_and_it_only_ever_falls(self):
        altitudes = np.radians(np.array([90, 60, 30, 10, 1]))
        assert np.all(np.diff(airmass.transmittance(0.145, altitudes)) < 0)


class TestSlabRadiance:
    def test_a_thin_column_is_linear_in_its_depth(self):
        """ Which is the approximation this replaces, and it is a good one while tau is small. """
        for tau in (1e-4, 1e-3, 1e-2):
            assert airmass.slab_radiance(1.0, tau) == pytest.approx(tau, rel=tau)

    def test_a_thick_column_saturates_at_the_source_function(self):
        assert airmass.slab_radiance(1.0, 20.0) == pytest.approx(1.0, abs=1e-8)
        assert airmass.slab_radiance(1.0, 1e6) <= 1.0

    def test_the_linear_form_is_five_times_high_at_the_horizon(self):
        """
        The measurement that decided the form: with k = 0.145 the zenith is 7% out and the horizon a
        factor of 5.1, and a wide-field camera keeps most of its pixels near the horizon.
        """
        tau = airmass.optical_depth(0.145, airmass.kasten_young(0.0))
        assert tau / airmass.slab_radiance(1.0, tau) == pytest.approx(5.1, rel=0.02)

        zenith = airmass.optical_depth(0.145, 1.0)
        assert zenith / airmass.slab_radiance(1.0, zenith) == pytest.approx(1.07, rel=0.02)
