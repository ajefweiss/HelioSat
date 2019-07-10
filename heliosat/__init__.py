# -*- coding: utf-8 -*-

from .spacecraft import DSCOVR, MES, STA, STB, VEX, WIND  # noqa: F401
from .spice import SpiceObject
from .util import configure_logging  # noqa: F401


__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.0.0"

_kernels_available = None
_kernels_loaded = None

_spacecraft_available = None


# common objects
Sun = SpiceObject(None, "SUN")
Mercury = SpiceObject(None, "MERCURY")
Venus = SpiceObject(None, "VENUS")
Earth = SpiceObject(None, "EARTH")
Moon = SpiceObject(None, "MOON")
Mars = SpiceObject(None, "MARS BARYCENTER")
Jupiter = SpiceObject(None, "JUPITER BARYCENTER")
Saturn = SpiceObject(None, "SATURN BARYCENTER")
Uranus = SpiceObject(None, "URANUS BARYCENTER")
Neptune = SpiceObject(None, "NEPTUNE BARYCENTER")
