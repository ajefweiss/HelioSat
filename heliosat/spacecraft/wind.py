# -*- coding: utf-8 -*-

"""wind.py

Custom trajectory function for Wind so that the trajectory is computed from the given CDF files.
"""

import datetime
import heliosat
import logging
import numpy as  np
import spiceypy

from heliosat.transform import transform_reference_frame
from heliosat.util import sanitize_dt
from typing import Sequence, Union


def wind_trajectory(self, dt: Union[datetime.datetime, Sequence[datetime.datetime]],
                   reference_frame: str = "J2000", observer: str = "SUN", units: str = "AU") -> np.ndarray:  # type: ignore
        logger = logging.getLogger(__name__)

        dt = sanitize_dt(dt)

        traj_t, traj_p = self.get(dt, "wind_trajectory", use_cache=True)

        traj_p[:, 0] *= -1
        traj_p[:, 1] *= -1

        traj_p += heliosat.Earth().trajectory(dt, units="km", reference_frame="HEE")
        
        traj = transform_reference_frame(dt, traj_p, "HEE", reference_frame)

        if units == "AU":
            traj *= 6.68459e-9
        elif units == "m":
            traj *= 1e3
        elif units == "km":
            pass
        else:
            logger.exception("unit \"%s\" is not supported", units)
            raise ValueError("unit \"{0!s}\" is not supported".format(units))

        return traj

# overwrite class function
heliosat.WIND.trajectory = wind_trajectory
