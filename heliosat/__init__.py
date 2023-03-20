# -*- coding: utf-8 -*-

import heliosat

from .spacecraft import Body, Spacecraft
from .spice import SpiceKernelManager as _SpiceKernelManager

__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.8.3"


_skm = _SpiceKernelManager()


# define body functions for common solar system objects
common_bodies = [
    ("Sun", "SUN"),
    ("Mercury", "MERCURY"),
    ("Venus", "VENUS"),
    ("Earth", "EARTH"),
    ("Moon", "MOON"),
    ("Mars", "MARS BARYCENTER"),
    ("Jupiter", "JUPITER BARYCENTER"),
    ("Saturn", "SATURN BARYCENTER"),
    ("Uranus", "URANUS BARYCENTER"),
    ("Neptune", "NEPTUNE BARYCENTER"),
]

for (_, __) in common_bodies:
    setattr(heliosat, _, Body(_, __))

# legacy support
Spacecraft.get_data = Spacecraft.get
