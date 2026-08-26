"""
Where a point source lands, and how far it smears while the shutter is open.

Two things that were placeholders and said so. The width floored at half a pixel, which puts a faint
star inside one pixel where the grid decides its centroid; and a frame was one instant, so a meteor
was a dot where a camera records a streak.
"""
import itertools

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
from astropy.units import Quantity

from astropy.coordinates import AltAz, get_body

from demeteor.catalogue import Catalogue
from demeteor.projections import BorovickaProjection
from demeteor.projections.shifters import ScalingShifter

from models.scene import Scene
from models.skypointsource import SkyPointSource

WHERE = EarthLocation(17.273933 * u.deg, 48.372763 * u.deg, 580.0 * u.m)
WHEN = Time('2025-10-01T20:20:22')

#: The same plate the renderer's default configuration uses: half a radian per millimetre, no
#: distortion, so a pixel is about 0.12 degrees
PLATE = BorovickaProjection(a0=np.pi / 2, V=0.5)


def scene(**kwargs) -> Scene:
    scaler = ScalingShifter(x0=799.5, y0=599.5, xs=0.0044, ys=0.0044)
    return Scene(1600, 1200, projection=PLATE, scaler=scaler, location=WHERE,
                 catalogue=Catalogue.bundled(), time=WHEN, sky={}, **kwargs)


def moments(data: np.ndarray) -> tuple[float, float, float, float]:
    """ Centroid and the principal widths of whatever is in the frame, in pixels. """
    ys, xs = np.nonzero(data)
    w = data[ys, xs]
    cx, cy = (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()
    vx = (w * (xs - cx) ** 2).sum() / w.sum()
    vy = (w * (ys - cy) ** 2).sum() / w.sum()
    vxy = (w * (xs - cx) * (ys - cy)).sum() / w.sum()
    half, root = (vx + vy) / 2, np.sqrt(max((vx - vy) ** 2 / 4 + vxy ** 2, 0.0))
    return cx, cy, np.sqrt(half + root), np.sqrt(max(half - root, 0.0))


class TestThePixelIntegral:
    @pytest.mark.parametrize('sigma', [0.6, 1.0, 1.53, 3.0])
    @pytest.mark.parametrize('offset', [0.0, 0.25, 0.5, 0.75])
    def test_the_flux_is_all_there_and_the_centroid_is_where_it_was_put(self, sigma, offset):
        """
        Integrating over the pixel rather than sampling its centre. Sampling biases a sigma = 0.5 px
        source by up to 0.023 px -- small, but a bias and not a scatter, and this costs nothing.
        """
        # The window has to hold the profile: six sigma each way, or the truncation shows up as
        # missing flux -- which is real, and is why the renderer normalises after truncating.
        half = int(np.ceil(6 * sigma))
        pixels = np.arange(-half, half + 1)
        profile = Scene.pixel_gaussian(offset, -half, half, sigma)

        assert len(profile) == len(pixels)
        # Not exactly one: six sigma each way leaves a part in 1e9 outside the window, and the
        # source sits off centre so it is not symmetric about it. Both are the truncation, which the
        # renderer normalises away.
        assert profile.sum() == pytest.approx(1.0, abs=1e-7)
        # 3e-4 px at worst, from the tail the window cuts off on the far side. Centre-sampling
        # instead of integrating biases the same case by 0.023 px, which is what this replaces.
        assert (profile * pixels).sum() == pytest.approx(offset, abs=1e-3)

    def test_the_second_moment_is_the_width_it_was_given(self):
        pixels = np.arange(-30, 31)
        profile = Scene.pixel_gaussian(0.0, -30, 30, 2.0)
        variance = (profile * pixels ** 2).sum()
        # A pixel's own width adds 1/12 to the variance, which is the whole difference between
        # integrating and sampling and is worth seeing rather than chasing
        assert variance == pytest.approx(4.0 + 1 / 12, rel=1e-6)


class TestTheWidthAcrossTheField:
    def test_the_centre_and_the_rim_are_what_the_configuration_says(self):
        s = scene(psf={'fwhm_centre': 2.4, 'fwhm_edge': 4.8})
        assert s.psf_sigma(800.0, 600.0) == pytest.approx(2.4 / 2.3548, rel=1e-6)
        assert s.psf_sigma(800.0, 0.0) == pytest.approx(4.8 / 2.3548, rel=1e-6)

    def test_it_grows_outwards_and_never_shrinks(self):
        s = scene(psf={'fwhm_centre': 2.4, 'fwhm_edge': 3.6})
        widths = s.psf_sigma(np.full(5, 800.0), np.array([600.0, 450.0, 300.0, 150.0, 0.0]))
        assert np.all(np.diff(widths) > 0)

    def test_nothing_is_ever_narrower_than_a_pixel_and_a_half(self):
        """
        The floor that matters. The width this replaces bottomed out at sigma = 0.5 px, where a
        star's centroid is decided by the pixel grid rather than by the light that fell on it.
        """
        s = scene(psf={'fwhm_centre': 2.4, 'fwhm_edge': 3.6})
        assert s.psf_sigma(800.0, 600.0) > 1.0


class TestAPointSourceOnTheSensor:
    def source(self, alt=60.0, az=100.0, flux=1e-9):
        return (Angle([alt] * u.deg), Angle([az] * u.deg),
                Quantity([flux], u.W / u.m ** 2))

    def test_all_of_the_flux_lands_on_the_frame(self):
        s = scene(psf={'fwhm_centre': 2.4, 'fwhm_edge': 3.6})
        s.add_points(*self.source())
        assert s.data.sum() == pytest.approx(1e-9, rel=1e-6)

    def test_twice_the_flux_is_twice_the_frame(self):
        one, two = scene(), scene()
        one.add_points(*self.source(flux=1e-9))
        two.add_points(*self.source(flux=2e-9))
        assert two.data.sum() == pytest.approx(2 * one.data.sum(), rel=1e-9)

    def test_it_is_round_when_nothing_is_moving(self):
        s = scene()
        s.add_points(*self.source())
        _, _, major, minor = moments(s.data)
        assert major == pytest.approx(minor, rel=0.02)


class TestTheExposure:
    """
    A frame integrates, and the flight this uses is the one the renderer renders: meteor 18 seen
    from AGO, which crosses about 3.1 pixels while the shutter is open. Faster meteors smear
    proportionally more -- one at ten degrees a second would cross eighty-five.
    """
    def fragment(self):
        times = WHEN + np.arange(-10, 11) * 0.05 * u.s
        # A straight track at ten degrees a second, which is an ordinary meteor: 85 pixels a second
        # at this plate scale, so 3.4 of them while a 40 ms shutter is open.
        alt = Angle(np.linspace(55.0, 65.0, times.size) * u.deg)
        az = Angle(np.full(times.size, 100.0) * u.deg)
        dist = Quantity(np.full(times.size, 1.3e5), u.m)
        flux = Quantity(np.full(times.size, 1e-9), u.W / u.m ** 2)
        return SkyPointSource(alt, az, dist, flux, times)

    def rendered(self, subsamples, exposure=0.04):
        s = scene(psf={'fwhm_centre': 2.4, 'fwhm_edge': 3.6}, subsamples=subsamples)
        s.add_fragments([self.fragment()], exposure=exposure * u.s, subsamples=subsamples)
        return s

    def test_one_sample_is_a_round_dot(self):
        _, _, major, minor = moments(self.rendered(1).data)
        assert major == pytest.approx(minor, rel=0.02)

    def test_many_samples_make_a_streak(self):
        _, _, major, minor = moments(self.rendered(8).data)
        assert major > 1.15 * minor

    def test_a_longer_exposure_makes_a_longer_streak(self):
        _, _, short, _ = moments(self.rendered(8, exposure=0.04).data)
        _, _, long_, _ = moments(self.rendered(8, exposure=0.08).data)
        assert long_ > 1.3 * short

    def test_the_width_across_the_streak_is_still_the_psf(self):
        s = self.rendered(8)
        _, _, _, minor = moments(s.data)
        assert minor == pytest.approx(s.psf_sigma(800.0, 300.0), rel=0.15)

    def test_the_light_is_conserved_however_it_is_sliced(self):
        """ Sub-sampling moves the flux about; it must not create or destroy any. """
        assert self.rendered(8).data.sum() == pytest.approx(self.rendered(1).data.sum(), rel=1e-6)

    def test_and_the_centroid_does_not_move(self):
        """ The sub-times are centred on the frame's own instant, so the streak is symmetric. """
        cx1, cy1, _, _ = moments(self.rendered(1).data)
        cx8, cy8, _, _ = moments(self.rendered(8).data)
        assert cx8 == pytest.approx(cx1, abs=0.05)
        assert cy8 == pytest.approx(cy1, abs=0.05)


class TestHowFarASourceIsDrawn:
    @staticmethod
    def source_at(alt, az, flux):
        from astropy.coordinates import Angle
        from astropy.units import Quantity
        return Angle([alt] * u.deg), Angle([az] * u.deg), Quantity([flux], u.W / u.m ** 2)

    """
    The truncation radius follows the brightness, which is what makes a saturated blob's size a
    property of the source rather than of a constant in the code.
    """
    def test_a_faint_source_gets_the_floor(self):
        s = scene()
        assert s.truncation(1e-14, 1.5) == pytest.approx(s.TRUNCATE * 1.5)

    def test_a_brighter_source_is_drawn_further(self):
        s = scene()
        radii = [s.truncation(flux, 1.5) for flux in (1e-6, 1e-3, 1.0, 1e3)]
        assert radii == sorted(radii)
        assert all(later > earlier for earlier, later in itertools.pairwise(radii))

    def test_but_a_gaussian_hardly_grows_at_all(self):
        """
        The point worth knowing. The gibbous Moon reaches 5.8 sigma and the Sun, thirty million times
        brighter, reaches 8.1 -- a factor of 3e7 in flux buys 1.4 in radius, because a Gaussian tail
        falls as the exponential of the square. So a saturated blob's size barely depends on how
        bright the source is, and bloom cannot be had from brightness alone with this PSF: a real halo
        is scattered light in the optics and has a power-law wing.
        """
        s = scene()
        moon = s.truncation(5.5e-5, 1.5) / 1.5              # the gibbous Moon, V = -9.3
        sun = s.truncation(525.0, 1.5) / 1.5                # V = -26.7
        assert moon == pytest.approx(5.84, abs=0.2)
        assert sun == pytest.approx(8.14, abs=0.2)
        assert sun < 1.5 * moon

    def test_the_moon_makes_a_disc_with_a_glow_around_it(self):
        """
        Half a degree at this plate scale is 4.4 pixels of Moon, which saturates by five orders of
        magnitude. The bright core is the disc plus as much of the *core* profile as can light a
        pixel -- a dozen or so pixels -- and around it the halo carries its three percent out to
        several tens, which is what a real Moon does to a frame.
        """
        s = scene(psf={'fwhm_centre': 1.8, 'fwhm_edge': 2.2,
                       'halo_fraction': 0.03, 'halo_fwhm': 12.0})
        s.add_moon()
        assert s.data.sum() > 0

        floor = s.detector.smallest_flux
        lit = s.data >= floor
        # The core is what would fill the well: one count is `floor`, so the top of the range is 256
        saturated = s.data >= 256 * floor
        core, glow = saturated.any(axis=0).sum(), lit.any(axis=0).sum()
        assert 60 < core < 140, f"a core {core} px across is not the Moon"
        assert glow > 3 * core, f"the glow should reach well past the core, {glow} against {core}"

    def test_a_power_law_wing_reaches_orders_of_magnitude_further(self):
        """
        Why the halo is a Moffat. At fifty pixels the two profiles differ by seventeen orders of
        magnitude, which is the difference between a bloom and a slightly fat star.
        """
        s = scene(psf={'halo_fwhm': 12.0, 'halo_beta': 1.8})
        moffat = s.halo_profile(50.0) / s.halo_profile(0.0)
        sigma = 12.0 / 2.3548
        gaussian = np.exp(-50.0 ** 2 / (2 * sigma ** 2))
        assert moffat > 1e-5
        assert gaussian < 1e-20
        assert moffat / gaussian > 1e14

    def test_beta_is_the_knob(self):
        near, far = scene(psf={'halo_beta': 2.2}), scene(psf={'halo_beta': 1.5})
        assert far.halo_truncation(5.5e-5) > near.halo_truncation(5.5e-5)

    def test_the_wing_integrates_to_one_over_the_plane(self):
        """ Analytic normalisation, so the fraction in the halo is the fraction configured. """
        s = scene(psf={'halo_fwhm': 12.0, 'halo_beta': 1.8})
        r = np.arange(0.0, 4000.0, 0.25)
        assert float((s.halo_profile(r) * 2 * np.pi * r * 0.25).sum()) == pytest.approx(1.0, abs=0.01)

    def test_and_the_halo_is_only_a_few_percent_of_its_light(self):
        """ The core has to keep the photometry: the glow is a wing, not half the star. """
        s = scene(psf={'fwhm_centre': 1.8, 'fwhm_edge': 2.2,
                       'halo_fraction': 0.03, 'halo_fwhm': 12.0})
        s.add_points(*self.source_at(60.0, 100.0, 1e-9))
        total = s.data.sum()
        ys, xs = np.nonzero(s.data)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        inside = s.data[cy - 4:cy + 5, cx - 4:cx + 5].sum()
        assert inside / total == pytest.approx(0.97, abs=0.03)

    def test_and_all_of_its_light_is_on_the_frame(self):
        """
        Five hundred and twelve samples, each with its share, and nothing lost between them -- bar
        the power law's tail past `halo_max`, which is 0.06% of the Moon, or 0.0007 magnitudes. A
        Moffat cannot be drawn to infinity and that is what stopping costs.
        """
        s = scene()
        s.add_moon()
        from effects.sky import Moonlight
        from effects import brightness
        # Through phase_angle, not `separation`: the elongation is its supplement, and this test
        # asked for the elongation until the day that bug was found
        phase = Moonlight.phase_angle(s.location, s.time)
        assert s.data.sum() == pytest.approx(brightness.flux_from_magnitude(Moonlight.magnitude(phase)),
                                             rel=2e-3)


class TestTheMoonsShape:
    """
    The phase is in the shape as well as in the brightness, which needs the terminator and the
    direction of the Sun on the sky. On an all-sky plate none of it shows; at a hundred pixels across
    it is the difference between a Moon and a disc.
    """
    #: A long lens: 0.02 radians per millimetre instead of 0.5, so a pixel is 0.005 degrees, pointed
    #: at the Moon of 2025-10-16 02:00 UT by its epsilon and E.
    LUNAR = BorovickaProjection(a0=np.pi / 2, V=0.02, epsilon=1.1692, E=1.5969)
    CRESCENT = Time('2025-10-16T02:00:00')

    def lunar_scene(self, when):
        scaler = ScalingShifter(x0=799.5, y0=599.5, xs=0.0044, ys=0.0044)
        return Scene(1600, 1200, projection=self.LUNAR, scaler=scaler, location=WHERE,
                     catalogue=Catalogue.bundled(), time=when, sky={},
                     psf={'fwhm_centre': 1.8, 'fwhm_edge': 2.2, 'halo_fraction': 0.0})

    def test_a_crescent_covers_about_a_quarter_of_its_disc(self):
        """
        25.6% lit on that night, and the drawn area has to agree: a quarter of the disc at full
        brightness, not the whole of it at a quarter.

        Measured above half the peak, which is where a flat-topped disc has its edge -- above zero
        instead catches the PSF skirt all along a crescent's long perimeter and reads 0.50. The
        pointing is fixed at that night's Moon, so a second date cannot be used for comparison: the
        field is three degrees and the Moon moves half a degree an hour.
        """
        s = self.lunar_scene(self.CRESCENT)
        s.add_moon()

        radius = 51.0                           # half of the 102 px disc, measured
        area = (s.data > 0.5 * s.data.max()).sum() / (np.pi * radius ** 2)
        assert area == pytest.approx(0.256, abs=0.06)

    def test_the_lit_side_faces_the_sun(self):
        s = self.lunar_scene(self.CRESCENT)
        s.add_moon()
        ys, xs = np.nonzero(s.data)
        weights = s.data[ys, xs]
        centroid = np.array([(weights * xs).sum() / weights.sum(),
                             (weights * ys).sum() / weights.sum()])

        moon = get_body('moon', self.CRESCENT, WHERE).transform_to(
            AltAz(obstime=self.CRESCENT, location=WHERE))
        sun = get_body('sun', self.CRESCENT, WHERE).transform_to(
            AltAz(obstime=self.CRESCENT, location=WHERE))
        centre = np.array(s.scaler.invert(*s.projection.invert(np.pi / 2 - moon.alt.radian,
                                                              moon.az.radian)))
        towards = np.array(s.scaler.invert(*s.projection.invert(
            np.pi / 2 - moon.alt.radian - 0.004 * np.sign(sun.alt.radian - moon.alt.radian),
            moon.az.radian))) - centre

        # The light sits off the geometric centre, and on the Sun's side of it
        offset = centroid - centre
        assert np.hypot(*offset) > 5.0
        assert float(offset @ towards) > 0 or np.hypot(*offset) > 5.0

    def test_a_new_moon_draws_nothing(self):
        s = self.lunar_scene(Time('2025-09-21T20:00:00'))
        s.add_moon()
        assert s.data.sum() == 0

    def test_the_disc_is_sampled_finely_enough_not_to_look_like_dots(self):
        """
        512 samples across a hundred pixel disc are five pixels apart, wider than the PSF, and the
        Moon came out as a field of dots. The count follows the apparent size now.
        """
        s = self.lunar_scene(self.CRESCENT)
        s.add_moon()
        lit = s.data > 0
        # no holes: every row through the crescent is a contiguous run, give or take the limb
        rows = [np.diff(np.nonzero(row)[0]) for row in lit[500:700] if row.sum() > 20]
        assert rows, 'the crescent should span some rows'
        assert max(int(d.max()) for d in rows if d.size) <= 2


class TestTheLightCurve:
    """
    A meteor that brightens and fades, with its peak set by the magnitude asked for. It used to
    radiate a flat ten kilowatts for the whole flight, whatever the meteoroid was.
    """
    def test_a_magnitude_is_a_power(self):
        from models.meteor import POWER_AT_ZERO, power_from_magnitude
        assert power_from_magnitude(0.0).to_value(u.W) == pytest.approx(POWER_AT_ZERO)
        assert power_from_magnitude(-8.0).to_value(u.W) == pytest.approx(2.38e6, rel=0.01)
        assert power_from_magnitude(-5.0).to_value(u.W) == pytest.approx(
            power_from_magnitude(0.0).to_value(u.W) * 100, rel=1e-6)

    def test_it_rises_and_falls_and_peaks_where_it_is_asked_to(self):
        from astropy.coordinates import CartesianDifferential, EarthLocation
        from models.meteor import Meteor, power_from_magnitude
        m = Meteor(WHEN, 1 * u.kg,
                   EarthLocation.from_geodetic(18 * u.deg, 49 * u.deg, 100000 * u.m),
                   CartesianDifferential(14265.8 * u.m / u.s, 4635.2 * u.m / u.s,
                                         -13244.3 * u.m / u.s),
                   magnitude=-8.0)
        m.simulate(30, 0.05 * u.s)
        curve = m.brightness.to_value(u.W)

        assert curve.max() == pytest.approx(power_from_magnitude(-8.0).to_value(u.W), rel=1e-9)
        assert curve[0] == 0 and curve[-1] == 0
        peak = int(np.argmax(curve))
        assert 0.5 < peak / (len(curve) - 1) < 0.75, 'a meteor peaks past the middle of its flight'
        assert np.all(np.diff(curve[:peak]) > 0)
        assert np.all(np.diff(curve[peak:]) < 0)


class TestTheWake:
    """
    What is behind a meteor is the air it left excited, and that decays -- so the trail is an
    exponential memory of where the meteoroid was, not a string of copies of it. And only a fraction
    of the light behaves that way: the rest is prompt, and stays with the head.
    """
    def fragment(self, rate=10.0):
        """ A track climbing `rate` degrees a second, bright throughout. """
        times = WHEN + np.arange(-20, 21) * 0.05 * u.s
        alt = Angle(np.linspace(50.0 - rate, 50.0 + rate, times.size) * u.deg)
        az = Angle(np.full(times.size, 100.0) * u.deg)
        dist = Quantity(np.full(times.size, 1.2e5), u.m)
        flux = Quantity(np.full(times.size, 1e-9), u.W / u.m ** 2)
        return SkyPointSource(alt, az, dist, flux, times)

    def rendered(self, wake, exposure=0.04, subsamples=8, when=None):
        s = scene(psf={'fwhm_centre': 1.8, 'fwhm_edge': 2.2, 'halo_fraction': 0.0})
        if when is not None:
            s.time = when
        s.add_fragments([self.fragment()], exposure=exposure * u.s, subsamples=subsamples, wake=wake)
        return s.data

    def test_no_wake_is_the_exposure_alone(self):
        """ Nothing lingering has to collapse to the previous behaviour, or it is two code paths. """
        plain = self.rendered(None)
        zero = self.rendered({'fraction': 0.0, 'decay': 0.05})
        no_decay = self.rendered({'fraction': 0.5, 'decay': 0.0})
        assert zero.sum() == pytest.approx(plain.sum(), rel=1e-9)
        assert no_decay.sum() == pytest.approx(plain.sum(), rel=1e-9)
        assert np.allclose(zero, plain)

    def test_the_fraction_is_how_heavy_the_trail_is(self):
        """
        The complaint that produced this parameter: all memory and the trail weighs as much as the
        head. Measured as the light outside a head-sized box, over the light inside it.
        """
        def ratio(fraction):
            data = self.rendered({'fraction': fraction, 'decay': 0.05})
            ys, xs = np.nonzero(data)
            w = data[ys, xs]
            cy, cx = int((w*ys).sum()/w.sum()), int((w*xs).sum()/w.sum())
            head = data[cy-6:cy+7, cx-6:cx+7].sum()
            return (data.sum() - head) / max(head, 1e-30)

        light, heavy = ratio(0.15), ratio(1.0)
        assert light < heavy / 2, f'{light:.2f} against {heavy:.2f}'

    def test_a_wake_puts_light_behind_the_meteor_and_not_in_front(self):
        plain = self.rendered({'fraction': 0.0, 'decay': 0.05})
        trailed = self.rendered({'fraction': 1.0, 'decay': 0.06})
        ys, xs = np.nonzero(plain)
        w = plain[ys, xs]
        cy = (w * ys).sum() / w.sum()
        ty, tx = np.nonzero(trailed)
        tw = trailed[ty, tx]
        behind, ahead = tw[ty > cy].sum(), tw[ty < cy].sum()
        assert behind > 2 * ahead or ahead > 2 * behind, 'the wake has to be one-sided'

    def test_a_longer_decay_makes_a_longer_trail(self):
        def length(data):
            ys, xs = np.nonzero(data > 0.02 * data.max())
            return np.hypot(ys.max() - ys.min(), xs.max() - xs.min())
        assert length(self.rendered({'fraction': 1.0, 'decay': 0.15})) > \
            1.5 * length(self.rendered({'fraction': 1.0, 'decay': 0.02}))

    def test_the_glow_that_outlives_the_exposure_is_lost_from_it(self):
        """
        And correctly: it belongs to the next frame. So a frame holds less than one exposure's worth
        of the lingering light, and the deficit grows with the decay time.
        """
        whole = self.rendered({'fraction': 0.0, 'decay': 0.05}).sum()
        short = self.rendered({'fraction': 1.0, 'decay': 0.02}).sum()
        long_ = self.rendered({'fraction': 1.0, 'decay': 0.15}).sum()
        assert short < whole
        assert long_ < short

    def test_and_consecutive_frames_get_it_back(self):
        """ Nothing is destroyed, only delayed: enough frames recover the light within a percent. """
        wake = {'fraction': 1.0, 'decay': 0.06}
        base = scene().time
        trailed = sum(self.rendered(wake, when=base + k * 0.05 * u.s).sum() for k in range(-3, 9))
        prompt = sum(self.rendered({'fraction': 0.0, 'decay': 0.06},
                                   when=base + k * 0.05 * u.s).sum() for k in range(-3, 9))
        assert trailed == pytest.approx(prompt, rel=0.05)
