# -*- coding: utf-8 -*-

from .spacecraft import DSCOVR, MES, PSP, STA, STB, VEX, WIND  # noqa: F401
from .spice import SpiceObject as _SpiceObject
from .util import get_heliosat_paths


__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.3.1"

_paths = get_heliosat_paths()
_spice = None

# common solar system objects
Sun = lambda: _SpiceObject(None, "SUN")  # noqa: E731
Mercury = lambda: _SpiceObject(None, "MERCURY")  # noqa: E731
Venus = lambda: _SpiceObject(None, "VENUS")  # noqa: E731
Earth = lambda: _SpiceObject(None, "EARTH")  # noqa: E731
Moon = lambda: _SpiceObject(None, "MOON")  # noqa: E731
Mars = lambda: _SpiceObject(None, "MARS BARYCENTER")  # noqa: E731
Jupiter = lambda: _SpiceObject(None, "JUPITER BARYCENTER")  # noqa: E731
Saturn = lambda: _SpiceObject(None, "SATURN BARYCENTER")  # noqa: E731
Uranus = lambda: _SpiceObject(None, "URANUS BARYCENTER")  # noqa: E731
Neptune = lambda: _SpiceObject(None, "NEPTUNE BARYCENTER")  # noqa: E731
