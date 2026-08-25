import datetime
import itertools
import math
import logging
import yaml
import numpy as np

from typing import TextIO, Any

import astropy.units as u
from demeteor.projections import Projection
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from enschema import Schema

log = logging.getLogger('root')
u.Wm2 = u.W / u.m**2


class SkyPointSource:
    """
    A point source moving in the sky, defined as the brightness
    of the source in the alt-az system at a specified time, I(alt, az, t)
    as a sequence of positions, times and brightnesses.
    Can be interpolated to any specified time,
    or extrapolated (in which case the brightness is assumed to be zero).
    """

    # `time` either way round. as_dict() writes time.iso, a string, so a file this class wrote
    # comes back as a string and a schema demanding a datetime rejects it -- which is what stopped
    # the renderer from reading anything the observer produced. Unquoted in the YAML it would be a
    # datetime instead, and astropy's Time takes either, so both are allowed rather than one
    # being converted on the way past.
    _schema = Schema({
        int: {
            'alt': float,
            'az': float,
            'dist': float,
            'i': float,
            'time': lambda value: isinstance(value, (str, datetime.datetime)),
        }
    })

    def __init__(self,
                 alt: Angle,            # Angle
                 az: Angle,             # Angle
                 dist: Quantity[u.m],
                 intensity: Quantity,   # Watt per square metre
                 time: Time):
        assert alt.shape == az.shape == dist.shape == intensity.shape == time.shape, \
            f"Shapes do not match: {alt.shape=}, {az.shape=}, {dist.shape=}, {intensity.shape=}, {time.shape=}"

        self._alt = alt
        self._az = az
        self._dist = dist
        self._intensity = intensity
        self._time = time

        assert self._intensity.unit.is_equivalent(u.W / u.m**2)
        assert self._dist.unit.is_equivalent(u.m)

    @property
    def alt(self):
        return self._alt

    @property
    def az(self):
        return self._az

    @property
    def dist(self):
        return self._dist

    @property
    def intensity(self):
        return self._intensity

    @property
    def time(self):
        return self._time

    def at_time(self, time: Time) -> tuple[Angle, Angle, Quantity]:
        """
        Return positions and intensities at time, as an (inter|extra)polation.
        """
        time = time.jd2
        stime = self.time.jd2
        alt = np.interp(time, stime, self.alt)
        az = np.interp(time, stime, self.az, period=math.tau)
        dist = np.interp(time, stime, self.dist)
        intensity = np.interp(time, stime, self.intensity, left=0, right=0)
        return alt, az, intensity

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return rf"<SkyPointSource with {len(self.alt)} frames at {self.time[0]}>"

    def as_dict(self):
        return {
            index: {
                'time': time.iso,
                'alt': float(alt.to(u.deg).value),
                'az': float(az.to(u.deg).value),
                'dist': float(dist.to(u.m).value),
                'i': float(intensity.to(u.Wm2).value),
            }
            for index, time, alt, az, dist, intensity
            in zip(itertools.count(), self.time, self.alt, self.az, self.dist, self.intensity)
        }

    @classmethod
    def load_dict(cls, data: dict[str, Any]):
        data = cls._schema.validate(data)
        time = Time([frame['time'] for index, frame in data.items()])
        alt = Angle([frame['alt'] * u.deg for index, frame in data.items()])
        az = Angle([frame['az'] * u.deg for index, frame in data.items()])
        dist = Quantity([frame['dist'] * u.m for index, frame in data.items()])
        inten = Quantity([frame['i'] * u.Wm2 for index, frame in data.items()])
        return SkyPointSource(alt, az, dist, inten, time)
