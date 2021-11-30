# -*- coding: utf-8 -*-

from .spacecraft import Body
from .spice import SpiceKernelManager as _SpiceKernelManager
from .util import configure_logging  # noqa: F401

__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.6.2"


_skm = _SpiceKernelManager()

# common solar system objects
Sun = lambda: Body("Sun", "SUN")
Mercury = lambda: Body("Mercury", "MERCURY")
Venus = lambda: Body("Venus", "VENUS")
Earth = lambda: Body("Earth", "EARTH")
Moon = lambda: Body("Moon", "MOON")
Mars = lambda: Body("Mars", "MARS BARYCENTER")
Jupiter = lambda: Body("Jupiter", "JUPITER BARYCENTER")
Saturn = lambda: Body("Saturn", "SATURN BARYCENTER")
Uranus = lambda: Body("Uranus", "URANUS BARYCENTER")
Neptune = lambda: Body("Neptune", "NEPTUNE BARYCENTER")
