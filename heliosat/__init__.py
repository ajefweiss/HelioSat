# -*- coding: utf-8 -*-

from .spacecraft import Body
from .spice import SpiceKernelManager as _SpiceKernelManager

__author__ = "Andreas J. Weiss"
__copyright__ = "Copyright (C) 2019 Andreas J. Weiss"
__license__ = "MIT"
__version__ = "0.8.2"


_skm = _SpiceKernelManager()


# common solar system objects
def Sun() -> Body:
    return Body("Sun", "SUN")


def Mercury() -> Body:
    return Body("Mercury", "MERCURY")


def Venus() -> Body:
    return Body("Venus", "VENUS")


def Earth() -> Body:
    return Body("Earth", "EARTH")


def Moon() -> Body:
    return Body("Moon", "MOON")


def Mars() -> Body:
    return Body("Mars", "MARS BARYCENTER")


def Jupiter() -> Body:
    return Body("Jupiter", "JUPITER BARYCENTER")


def Saturn() -> Body:
    return Body("Saturn", "SATURN BARYCENTER")


def Uranus() -> Body:
    return Body("Uranus", "URANUS BARYCENTER")


def Neptune() -> Body:
    return Body("Neptune", "NEPTUNE BARYCENTER")
