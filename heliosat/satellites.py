# -*- coding: utf-8 -*-

"""satellites.py

Implements spacecraft classes.
"""

import datetime
import logging
import heliosat
import inspect
import spiceypy

from heliosat.coordinates import transform_pos
from heliosat.spacecraft import Spacecraft
from typing import Iterable, List, Optional, Union


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

    def trajectory(self, t: Union[datetime.datetime, Iterable[datetime.datetime]],
                   frame: str = "J2000", observer: str = "SUN", units: str = "AU"):
        """Evaluate body trajectory at given datetimes.

        Parameters
        ----------
        t : Union[datetime.datetime, Iterable[datetime.datetime]]
            Evaluate datetime(s).
        frame : str, optional
            Trajectory reference frame, by default "J2000".
        observer : str, optional
            Observer body name, by default "SUN".
        units : str, optional
            Output units, by default "AU".

        Returns
        -------
        np.ndarray
            Body trajectory.

        Raises
        ------
        NotImplementedError
            If units are invalid.
        """
        logger = logging.getLogger(__name__)

        if heliosat._spice is None:
            logger.info("running spice_init")
            spice_init(False)

        if isinstance(t, datetime.datetime):
            traj_t, traj_p = self.get_data([t, t + datetime.timedelta(hours=1)], "wind_trajectory", cache=True)
            traj_p = traj_p[0]
            traj_p[0] *= -1
            traj_p[1] *= -1

            traj_p += heliosat.Earth().trajectory(t, units="km", frame="HEE")

            trajectory = spiceypy.mxv(spiceypy.pxform("HEE", frame, spiceypy.datetime2et(t)), traj_p)
        elif len(t) == 2:
            traj_t, traj_p = self.get_data([t[0], t[1], t[1] + datetime.timedelta(hours=1)], "wind_trajectory", cache=True)
            traj_p = traj_p[0:2]

            traj_p[:, 0] *= -1
            traj_p[:, 1] *= -1

            traj_p += heliosat.Earth().trajectory(t, units="km", frame="HEE")
            
            trajectory = transform_pos(t, traj_p, "HEE", frame)
        else:
            traj_t, traj_p = self.get_data(t, "wind_trajectory", cache=True)
            traj_p[:, 0] *= -1
            traj_p[:, 1] *= -1

            traj_p += heliosat.Earth().trajectory(t, units="km", frame="HEE")
            
            trajectory = transform_pos(t, traj_p, "HEE", frame)

        if units == "AU":
            trajectory *= 6.68459e-9
        elif units == "m":
            trajectory *= 1e3
        elif units == "km":
            pass
        else:
            logger.exception("unit \"%s\" is not supported", units)
            raise NotImplementedError("unit \"%s\" is not supported", units)

        return trajectory


def select_satellite(satellite: str) -> Spacecraft:
    if hasattr(heliosat, satellite):
        sat = getattr(heliosat, satellite)

        if Spacecraft in inspect.getmro(sat):
            return sat

    raise NotImplementedError("unkown satellite \"%s\"", satellite)
