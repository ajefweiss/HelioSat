# -*- coding: utf-8 -*-

from .spacecraft import Body
from .spice import SpiceKernelManager as _SpiceKernelManager

__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.8.2"


_skm = _SpiceKernelManager()


# common solar system objects
def Sun():
    return Body("Sun", "SUN")


def Mercury():
    return Body("Mercury", "MERCURY")


def Venus():
    return Body("Venus", "VENUS")


def Earth():
    return Body("Earth", "EARTH")


def Moon():
    return Body("Moon", "MOON")


def Mars():
    return Body("Mars", "MARS BARYCENTER")


def Jupiter():
    return Body("Jupiter", "JUPITER BARYCENTER")


def Saturn():
    return Body("Saturn", "SATURN BARYCENTER")


def Uranus():
    return Body("Uranus", "URANUS BARYCENTER")


def Neptune():
    return Body("Neptune", "NEPTUNE BARYCENTER")
