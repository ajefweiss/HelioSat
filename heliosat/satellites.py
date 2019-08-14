# -*- coding: utf-8 -*-

from heliosat.data import DataObject
from heliosat.spice import SpiceObject


class DSCOVR(SpiceObject, DataObject):
    def __init__(self):
        super(DSCOVR, self).__init__("dscovr", "EARTH")


class MES(SpiceObject, DataObject):
    def __init__(self):
        super(MES, self).__init__("messenger", "MESSENGER")


class PSP(SpiceObject, DataObject):
    def __init__(self):
        from spiceypy import boddef, bodn2c
        from spiceypy.utils.support_types import SpiceyError

        super(PSP, self).__init__("psp", "SPP")

        try:
            bodn2c("SPP_SPACECRAFT")
        except SpiceyError:
            # kernel fix
            boddef("SPP_SPACECRAFT", bodn2c("SPP"))


class STA(SpiceObject, DataObject):
    def __init__(self):
        super(STA, self).__init__("stereo_ahead", "STEREO AHEAD", "stereo")


class STB(SpiceObject, DataObject):
    def __init__(self):
        super(STB, self).__init__("stereo_behind", "STEREO BEHIND", "stereo")


class VEX(SpiceObject, DataObject):
    def __init__(self):
        super(VEX, self).__init__("venus_express", "VENUS EXPRESS")


class WIND(SpiceObject, DataObject):
    def __init__(self):
        super(WIND, self).__init__("wind", "EARTH")
