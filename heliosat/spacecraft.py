# -*- coding: utf-8 -*-

from heliosat.instruments import FaradayCup, Magnetometer
from heliosat.spice import SpiceObject


class DSCOVR(SpiceObject, FaradayCup, Magnetometer):
    def __init__(self):
        super(DSCOVR, self).__init__("dscovr", "EARTH")

    def get_fc_files(self, start_time, stop_time):
        """Get raw plasma data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw plasma data files
        """
        from heliosat.instruments import get_fc_files_dscovr

        return get_fc_files_dscovr(self, start_time, stop_time)

    def get_mag_files(self, start_time, stop_time):
        """Get raw magnetic field data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw magnetic field data files
        """
        from heliosat.instruments import get_mag_files_dscovr

        return get_mag_files_dscovr(self, start_time, stop_time)


class MES(SpiceObject, Magnetometer):
    def __init__(self):
        super(MES, self).__init__("messenger", "MESSENGER")

    def get_mag_files(self, start_time, stop_time):
        """Get raw magnetic field data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw magnetic field data files
        """
        from heliosat.instruments import get_mag_files_mes

        return get_mag_files_mes(self, start_time, stop_time)


class PSP(SpiceObject):
    def __init__(self):
        from spiceypy import boddef, bodn2c
        from spiceypy.utils.support_types import SpiceyError

        super(PSP, self).__init__("psp", "SPP")

        try:
            bodn2c("SPP_SPACECRAFT")
        except SpiceyError:
            # kernel fix
            boddef("SPP_SPACECRAFT", bodn2c("SPP"))


class STA(SpiceObject, FaradayCup, Magnetometer):
    def __init__(self):
        super(STA, self).__init__("stereo_ahead", "STEREO AHEAD", "stereo")

    def get_fc_files(self, start_time, stop_time):
        """Get raw plasma data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw plasma data files
        """
        from heliosat.instruments import get_fc_files_stereo

        return get_fc_files_stereo(self, start_time, stop_time)


class STB(SpiceObject, FaradayCup, Magnetometer):
    def __init__(self):
        super(STB, self).__init__("stereo_behind", "STEREO BEHIND", "stereo")

    def get_fc_files(self, start_time, stop_time):
        """Get raw plasma data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw plasma data files
        """
        from heliosat.instruments import get_fc_files_stereo

        return get_fc_files_stereo(self, start_time, stop_time)


class VEX(SpiceObject, Magnetometer):
    def __init__(self):
        super(VEX, self).__init__("venus_express", "VENUS EXPRESS")

    def get_mag_files(self, start_time, stop_time):
        """Get raw magnetic field data files. The files are downloaded if necessary.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time

        Returns
        -------
        list
            raw magnetic field data files
        """
        from heliosat.instruments import get_mag_files_vex

        return get_mag_files_vex(self, start_time, stop_time)


class WIND(SpiceObject, Magnetometer):
    def __init__(self):
        super(WIND, self).__init__("wind", "EARTH")
