# -*- coding: utf-8 -*-

"""dscovr.py

Custom trajectory functions where the trajectory is computed from the given CDF files.
"""

import datetime as dt
import logging as lg
from typing import Any, Optional, Sequence, Union

import numpy as np

import heliosat
from heliosat.routines import transform_reference_frame
from heliosat.smoothing import smooth_data
from heliosat.spacecraft import _generate_endpoints
from heliosat.util import dt_utc_from_ts, get_any, sanitize_dt


def dscovr_trajectory(
    self: object,
    dtp: Union[dt.datetime, Sequence[dt.datetime]],
    observer: str = "SUN",
    units: str = "AU",
    smoothing: Optional[str] = None,
    **kwargs: Any
) -> np.ndarray:
    dtp = sanitize_dt(dtp)

    sampling_freq = get_any(
        kwargs, ["sampling_freq", "sampling_frequency", "sampling_rate"], 3600
    )

    # use dt list as endpoints
    if kwargs.pop("as_endpoints", False):
        dtp = _generate_endpoints(dtp, sampling_freq)

    if smoothing:
        lg.info(
            "note that smoothing is implemented differently for dscovr trajectories (slower)"
        )

    traj_t, traj_p = self.get(
        dtp,
        "dscovr_trajectory",
        return_datetimes=False,
        cached=kwargs.get("cached", False),
    )
    reference_frame = get_any(kwargs, ["reference_frame", "frame"], "J2000")
    return_datetimes = kwargs.pop("return_datetimes", False)

    # dscovr has a quirk where it normally has unevenly spaced data points, this is a dirty fix
    if 10 * (traj_t[1] - traj_t[0]) < (traj_t[2] - traj_t[1]) and not kwargs.get(
        "nofix", False
    ):
        lg.info(
            'attempting to "fix" dscovr trajectory data, removing unevely spaced data points (disable nofix=True)',
        )
        traj_t = traj_t[::2]
        traj_p = traj_p[::2]

    traj_t, traj_p = smooth_data(
        dtp,
        traj_t,
        traj_p,
        smoothing=smoothing,
        smoothing_scale=kwargs.get("smoothing_scale", 7200),
    )

    traj_t = dt_utc_from_ts(traj_t)

    # convert int array to float
    traj_p = np.array(traj_p, dtype=float)

    if observer == "SUN":
        traj_p[:, 0] *= -1
        traj_p[:, 1] *= -1

        traj_p += heliosat.Earth.trajectory(traj_t, units="km", reference_frame="HEE")

        traj_p = transform_reference_frame(traj_t, traj_p, "HEE", reference_frame)
    elif observer.upper() == "EARTH":
        pass
    else:
        raise NotImplementedError("invalid observer: %s", observer)

    if units == "AU":
        traj_p *= 6.68459e-9
    elif units == "m":
        traj_p *= 1e3
    elif units == "km":
        pass
    else:
        raise ValueError('unit "{0!s}" is not supported'.format(units))

    if len(traj_p) == 1:
        if return_datetimes:
            return traj_t, traj_p[0]
        else:
            return traj_p[0]
    else:
        if return_datetimes:
            return traj_t, traj_p
        else:
            return traj_p


# overwrite class functions
heliosat.DSCOVR.trajectory = dscovr_trajectory
