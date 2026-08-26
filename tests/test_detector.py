"""
The readout chain, in electrons.

Every number here can be compared with a datasheet, which is the point of the chain replacing
`Scene.gain = 1e13` and a `lambda x: x * 10`. The statistics are what these tests are for: the stage
this replaces computed `20 * Poisson(40)`, which has the right mean and four hundred times the
variance, and nothing said so.
"""
import numpy as np
import pytest

from models.detector import Detector

#: A frame small enough to be fast and large enough for its statistics to mean something
SHAPE = (200, 300)


def frame(**kwargs) -> tuple[Detector, np.ndarray, np.ndarray]:
    settings = dict(aperture=4.3, qe=0.20, wavelength=550.0, exposure=0.04,
                    dark=5.0, read_noise=200.0, bias=500.0, full_well=128000.0, bits=8,
                    gain=10000.0, excess_noise=1.0, glow=0.0, glow_radius=70.0)
    settings.update(kwargs)
    xs, ys = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    return Detector(detector=settings), xs, ys


class TestPhotoelectrons:
    def test_a_zeroth_magnitude_star(self):
        """
        1.1e-8 W/m2 through a 4.3 mm pupil at 20% for 40 ms. Nothing subtle -- the point is that the
        number follows from four things one can measure rather than from a fitted constant.
        """
        d, _, _ = frame()
        assert d.photoelectrons(1.107e-8) == pytest.approx(3561, rel=0.01)

    def test_it_is_linear_in_flux_and_in_exposure(self):
        d, _, _ = frame()
        assert d.photoelectrons(2e-9) == pytest.approx(2 * d.photoelectrons(1e-9))
        slow, _, _ = frame(exposure=0.08)
        assert slow.photoelectrons(1e-9) == pytest.approx(2 * d.photoelectrons(1e-9))

    def test_a_wider_pupil_gathers_as_its_area(self):
        small, _, _ = frame(aperture=4.0)
        big, _, _ = frame(aperture=8.0)
        assert big.photoelectrons(1e-9) == pytest.approx(4 * small.photoelectrons(1e-9), rel=1e-9)


class TestNoiseStatistics:
    def test_dark_current_has_poisson_variance(self):
        """
        The test that would have caught `20 * Poisson(40)`: for a Poisson process the variance equals
        the mean, and no multiplier outside the draw can be allowed to change that.
        """
        # A well and a depth that resolve a thousand electrons: one count is 0.06 of one, so the
        # quantisation contributes nothing to the variance being measured.
        d, xs, ys = frame(dark=25000.0, gain=1.0, read_noise=0.0, bias=0.0,
                          full_well=4000.0, bits=16, excess_noise=1.0)
        electrons = d.readout(np.zeros(SHAPE), xs, ys) * d.full_well / (d.saturation + 1)
        expected = d.dark * d.exposure
        assert electrons.mean() == pytest.approx(expected, rel=0.02)
        assert electrons.var() == pytest.approx(expected, rel=0.10)

    def test_read_noise_is_the_configured_sigma(self):
        d, xs, ys = frame(dark=0.0, glow=0.0, read_noise=3000.0, bias=64000.0,
                          full_well=128000.0, bits=16)
        counts = d.readout(np.zeros(SHAPE), xs, ys)
        electrons = counts * d.full_well / (d.saturation + 1)
        assert electrons.mean() == pytest.approx(64000.0, rel=0.01)
        assert electrons.std() == pytest.approx(3000.0, rel=0.10)

    def test_the_intensifier_adds_variance_beyond_poisson(self):
        signal = np.full(SHAPE, 1e-11)
        plain, xs, ys = frame(excess_noise=1.0, read_noise=0.0, bias=0.0, dark=0.0, bits=16)
        noisy, _, _ = frame(excess_noise=2.5, read_noise=0.0, bias=0.0, dark=0.0, bits=16)
        assert noisy.readout(signal, xs, ys).var() > 1.5 * plain.readout(signal, xs, ys).var()

    def test_the_glow_is_brightest_in_the_middle(self):
        d, xs, ys = frame(glow=20.0, dark=0.0, read_noise=0.0, bias=0.0, bits=16)
        counts = d.readout(np.zeros(SHAPE), xs, ys)
        centre = counts[SHAPE[0] // 2 - 5:SHAPE[0] // 2 + 5, SHAPE[1] // 2 - 5:SHAPE[1] // 2 + 5]
        assert centre.mean() > 5 * counts[:10, :10].mean() + 1


class TestSaturation:
    def test_nothing_exceeds_the_output_depth(self):
        d, xs, ys = frame()
        counts = d.readout(np.full(SHAPE, 1e-6), xs, ys)
        assert counts.max() == d.saturation == 255
        assert counts.min() >= 0

    def test_a_bright_source_clips_flat(self):
        """
        Which is what blooming is, and why `Scene.intensity_to_sigma` is gone: a flat top comes from
        the full well, not from a wider Gaussian.
        """
        d, xs, ys = frame()
        counts = d.readout(np.full(SHAPE, 1e-7), xs, ys)
        assert (counts == d.saturation).mean() > 0.99

    def test_the_depth_follows_the_bits(self):
        assert frame(bits=8)[0].saturation == 255
        assert frame(bits=12)[0].saturation == 4095
        assert frame(bits=16)[0].saturation == 65535

    def test_one_photoelectron_is_worth_what_the_gain_and_the_well_say(self):
        """
        The ratio that decides how a frame looks: with these two, one photoelectron is twenty counts,
        which is what puts a dark sky near twenty-five.
        """
        d, _, _ = frame()
        assert d.gain / d.full_well * (d.saturation + 1) == pytest.approx(20.0, rel=1e-9)
