#!/usr/bin/env python
import argparse
import logging
import sys

import numpy as np

from multiprocessing import Pool

import yaml
from astropy.coordinates import EarthLocation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
from scalyca import Scalyca

from models.meteor import Meteor

log = logging.getLogger('root')
VERSION = '0.1.1'


class MeteorSimulatorCLI(Scalyca):
    _prog = 'Vamos meteor simulator'
    _version = VERSION

    def add_arguments(self):
        self.add_argument('meteor', type=argparse.FileType('r'),
                          help="YAML file with meteor configuration")
        self.add_argument('-o', '--outfile', type=argparse.FileType('w'), default=sys.stdout,
                          help="Output YAML file")

    def initialize(self):
        self.t0 = Time(self.config.start)
        self.dt = 0.05 * u.s
        self.times = self.t0 + np.arange(0, self.config.count) * self.dt

    def main(self):
        m = Meteor.create_from_yaml(yaml.safe_load(self.args.meteor))
        m.simulate(self.config.count, self.dt)

        if self.args.debug:
            log.debug(f"Meteoroid data:")
            for time, pos, br in zip(m.time, m.position, m.brightness):
                log.debug(
                    f"{time}, {pos.to_geodetic()}, {br}"
                )

        if self.args.outfile:
            yaml.dump(m.as_dict(), self.args.outfile)


def simulate(
        position: EarthLocation,
        velocity: CartesianDifferential,
        mass: u.Quantity[u.kg],
        time: Time,
    ):
    pool = Pool(self.config.cores)

# An application that runs at module scope runs on any import -- a test collecting it, a tool
# reading it, something reaching for one of its classes -- and what it does then is parse whatever
# argv that process happens to have and exit. See render.py for where that actually bit.
if __name__ == '__main__':
    MeteorSimulatorCLI().run()
