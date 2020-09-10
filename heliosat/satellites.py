# -*- coding: utf-8 -*-

"""satellites.py

Implements spacecraft classes.
"""

import spiceypy

from heliosat.spacecraft import Spacecraft


class BEPI(Spacecraft):
    def __init__(self, **kwargs):
        super(BEPI, self).__init__("bepi_colombo", body_name="BEPICOLOMBO MPO", **kwargs)


class DSCOVR(Spacecraft):
    def __init__(self, **kwargs):
        super(DSCOVR, self).__init__("dscovr", body_name="EARTH", **kwargs)


class MES(Spacecraft):
    def __init__(self, **kwargs):
        super(MES, self).__init__("messenger", body_name="MESSENGER", **kwargs)


class PSP(Spacecraft):
    def __init__(self, **kwargs):
        from spiceypy.utils.support_types import SpiceyError

        super(PSP, self).__init__("psp", body_name="SPP", **kwargs)

        try:
            spiceypy.bodn2c("SPP_SPACECRAFT")
        except SpiceyError:
            # kernel fix that is required due to PSP having different names
            spiceypy.boddef("SPP_SPACECRAFT", spiceypy.bodn2c("SPP"))


class SOLO(Spacecraft):
    def __init__(self, **kwargs):
        super(SOLO, self).__init__("solar_orbiter", body_name="SOLAR ORBITER", **kwargs)


class STA(Spacecraft):
    def __init__(self, **kwargs):
        super(STA, self).__init__("stereo_ahead", body_name="STEREO AHEAD", **kwargs)


class STB(Spacecraft):
    def __init__(self, **kwargs):
        super(STB, self).__init__("stereo_behind", body_name="STEREO BEHIND", **kwargs)


class VEX(Spacecraft):
    def __init__(self, **kwargs):
        super(VEX, self).__init__("venus_express", body_name="VENUS EXPRESS", **kwargs)


class WIND(Spacecraft):
    def __init__(self, **kwargs):
        super(WIND, self).__init__("wind", body_name="EARTH", **kwargs)


def select_satellite(satellite):
    if satellite.upper() == "DSCOVR":
        return DSCOVR()
    elif satellite.upper() == "MES":
        return MES()
    elif satellite.upper() == "PSP":
        return PSP()
    elif satellite.upper() == "STA":
        return STA()
    elif satellite.upper() == "STB":
        return STA()  
    elif satellite.upper() == "VEX":
        return VEX()
    elif satellite.upper() == "WIND":
        return WIND()
    else:
        raise NotImplementedError("unkown satellite \"%s\"", satellite.upper())
