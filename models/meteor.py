import itertools
import logging
from typing import TextIO, Any

import numpy as np

import astropy.units as u
import astropy.constants as const
import yaml
from astropy.coordinates import EarthLocation, CartesianRepresentation, CartesianDifferential
from astropy.time import Time

log = logging.getLogger('root')


#: Radiated power of a zero-magnitude meteor, in watts, at the hundred kilometres an absolute
#: magnitude is defined at. Ceplecha et al. (1998) and every photometric mass since; the band it
#: refers to is the visual one, which is also what `Scene.vmag_to_intensity` assumes, so the two ends
#: of the pipeline agree about what a magnitude means.
POWER_AT_ZERO = 1500.0

#: If a meteor config does not say. Ordinary rather than spectacular.
MAGNITUDE = -2.0


def power_from_magnitude(magnitude: float) -> u.Quantity:
    """
    Peak radiated power from an absolute (100 km) magnitude.

    -8 is 2.4 megawatts, -2 is 9.5 kilowatts, +2 is 240 watts. What an observer measures is fainter
    by `5 log10(d / 100 km)`, so the same fireball seen from 130 km reads -7.4.
    """
    return POWER_AT_ZERO * 10 ** (-0.4 * float(magnitude)) * u.W


class Meteor:
    def __init__(self,
                 initial_time: Time,
                 initial_mass: u.Quantity[u.kg],
                 initial_position: EarthLocation,
                 initial_velocity: CartesianDifferential,
                 initial_brightness: u.Quantity[u.W] = 0 * u.W,
                 magnitude: float = None):
        self.time: Time = initial_time
        self.mass: u.Quantity[u.kg] = initial_mass
        self.position: EarthLocation = initial_position
        self.velocity: CartesianDifferential = initial_velocity
        self.brightness: u.Quantity[u.watt] = initial_brightness
        #: What the light curve peaks at, from the absolute magnitude it is asked for
        self.magnitude = MAGNITUDE if magnitude is None else float(magnitude)
        self.peak_power = power_from_magnitude(self.magnitude)

    def simulate(self,
                 steps: int,
                 dt: u.Quantity[u.s]):
        n = 0
        pos = [self.position]
        vel = [self.velocity]

        while n < steps:
            cp = CartesianRepresentation(pos[-1].to_geocentric())
            cv = vel[-1]
            newpos = cp + cv * dt
            pos.append(EarthLocation.from_geocentric(newpos.x, newpos.y, newpos.z))
            vel.append(cv)

            n += 1

        self.position = EarthLocation.from_geocentric(
            u.Quantity([p.x for p in pos]),
            u.Quantity([p.y for p in pos]),
            u.Quantity([p.z for p in pos]),
        )
        self.velocity = CartesianDifferential(
            u.Quantity([v.d_x for v in vel]),
            u.Quantity([v.d_y for v in vel]),
            u.Quantity([v.d_z for v in vel]),
        )
        self.time = Time(
            self.time + np.arange(0, len(self.position)) * dt,
        )
        # The light curve, in normalised time: a rise and a fall, peaking at tau = 5/8. It was
        # `1e4 W` flat -- ten kilowatts for the whole flight, whatever the meteoroid was doing -- with
        # this shape sitting commented out beside it. A meteor that does not brighten and fade is not
        # a meteor, and the peak is now the configured magnitude rather than a constant in the code.
        tau = np.linspace(0, 1, steps + 1)
        shape = (1 - tau) ** 3 * tau ** 5
        self.brightness = self.peak_power * shape / shape.max()

    @staticmethod
    def acceleration(position: EarthLocation, velocity: CartesianDifferential) -> u.Quantity[u.N]:
        """
        Calculate the gravitational acceleration and inertial accelerations
        acting on a body at defined position and velocity.
        """

        pos = position.get_itrs().cartesian
        r = pos.norm()
        v = velocity.norm()
        omega = CartesianRepresentation(0 * u.rad / u.s, 0 * u.rad / u.s, 7.2921550e-5 * u.rad / u.s).xyz
        return (
            -((const.G * const.M_earth / r**3) * pos).xyz.T
            - 2 * np.cross(omega.to_value(u.rad / u.s), velocity.d_xyz.T.to_value(u.m / u.s)) * u.m / u.s**2
            - np.cross(omega, np.cross(omega, pos.xyz.T)) / (u.rad**2)
        ).to(u.m / u.s**2)

    def __str__(self):
        return f"<Meteor at {self.time[0]}>"

    def __repr__(self):
        return f"<Meteor at {self.time[0]}>"

    def as_dict(self):
        """
        Returns a dictionary representation of the meteor, suitable for saving.
        """
        return {
            index: {
                'time': time.iso,
                'pos': {
                    'lat': float(position.lat.value),
                    'lon': float(position.lon.value),
                    'alt': float(position.height.to(u.m).value),
                    'x': float(position.x.to(u.m).value),
                    'y': float(position.y.to(u.m).value),
                    'z': float(position.z.to(u.m).value),
                },
                'vel': {
                    'vx': float(velocity.d_x.value),
                    'vy': float(velocity.d_y.value),
                    'vz': float(velocity.d_z.value),
                },
                'i': float(brightness.to(u.W).value),
            }
            for index, time, position, velocity, brightness in
            zip(itertools.count(), self.time, self.position, self.velocity, self.brightness)
        }

    @staticmethod
    def create_from_yaml(data):
        return Meteor(
            Time(data['time']),
            data['mass'] * u.kg,
            EarthLocation.from_geodetic(
                data['location']['longitude'] * u.deg,
                data['location']['latitude'] * u.deg,
                data['location']['altitude'] * u.m,
            ),
            CartesianDifferential(
                data['velocity']['x'] * u.m / u.s,
                data['velocity']['y'] * u.m / u.s,
                data['velocity']['z'] * u.m / u.s,
            ),
            0 * u.W,
            magnitude=data.get('magnitude', MAGNITUDE),
        )

    @staticmethod
    def load_dict(data: dict[str, Any]):
        time = Time([frame['time'] for index, frame in data.items()])
        position = EarthLocation.from_geodetic(
            [frame['pos']['lon'] * u.deg for index, frame in data.items()],
            [frame['pos']['lat'] * u.deg for index, frame in data.items()],
            [frame['pos']['alt'] * u.m for index, frame in data.items()],
        )
        velocity = CartesianDifferential(
            [frame['vel']['vx'] for index, frame in data.items()] * u.m/u.s,
            [frame['vel']['vy'] for index, frame in data.items()] * u.m/u.s,
            [frame['vel']['vz'] for index, frame in data.items()] * u.m/u.s,
        )
        brightness = u.Quantity([frame['i'] for index, frame in data.items()]) * u.W
        return Meteor(time, 0 * u.kg, position, velocity, brightness)