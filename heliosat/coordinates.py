# -*- coding: utf-8 -*-

"""coordinates.py
Implements coordinate transformation functions requiring SPICE.
"""

import datetime
import heliosat
import multiprocessing
import numpy as np
import spiceypy

from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional, Union


def transform_lonlat(t: Union[datetime.datetime, Iterable[datetime.datetime]],
                     lonlat: np.ndarray, frame_from: str, frame_to: str) -> np.ndarray:
    """Transform longitude/latitude's from one reference frame to another.
    Inclination is not transformed, therefore the transformation only works in between reference
    frames that share the same z-axis.
    Parameters
    ----------
    t : Union[datetime.datetime, Iterable[datetime.datetime]]
        Evaluation datetimes.
    lonlat : np.ndarray
        Longitude/latitude's array.
    frame_from : str
        Source refernce frame.
    frame_to : str
        Target reference frame.
    Returns
    -------
    np.ndarray
        Transformed longitude/latitude's.
    Raises
    ------
    TypeError
        Variable t is not a valid datetime or Iterable of datetimes
    ValueError
        Source and target frame are equal.
    """
    if frame_from == frame_to:
        raise ValueError("source and target frame are equal")

    lons_rad = 2 * np.pi * lonlat[:, 0] / 360
    lats_rad = 2 * np.pi * lonlat[:, 1] / 360

    vecs = np.array([
        [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]
        for (lon, lat) in list(zip(lons_rad, lats_rad))
    ])

    if isinstance(t, datetime.datetime):
        for i in range(0, len(vecs)):
            vecs[i] = spiceypy.mxv(spiceypy.pxform(frame_from, frame_to,
                                   spiceypy.datetime2et(t)),
                                   vecs[i])
    elif hasattr(t, "__iter__"):
        for i in range(0, len(t)):
            vecs[i] = spiceypy.mxv(spiceypy.pxform(frame_from, frame_to,
                                   spiceypy.datetime2et(t[i])),
                                   vecs[i])
    else:
        raise TypeError("Variable t is not a valid datetime or Iterable of datetimes")

    return np.array([
        [360 * np.arccos(v[0]) / 2 / np.pi, 360 * np.arcsin(v[2]) / 2 / np.pi]
        for v in vecs
    ])


def transform_pos(t: Union[datetime.datetime, Iterable[datetime.datetime]],
                  data: np.ndarray, frame_from: str, frame_to: str,
                  frame_cadence: Optional[float] = None) -> np.ndarray:
    """Transform 3D coordinates from one reference frame to another.
    Parameters
    ----------
    t : Union[datetime.datetime, Iterable[datetime.datetime]]
        Evaluation datetimes.
    data : np.ndarray
        3D vector array.
    frame_from : str
        Source refernce frame.
    frame_to : str
        Target reference frame.
    frame_cadence: float, optional
        Evaluate frame transformation matrix every "frame_cadence" seconds instead of at very
        time point (significant speed up), by default None.
    Returns
    -------
    np.ndarray
        Transformed vector array in target reference frame.
    Raises
    ------
    ValueError
        Data array has the wrong dimensions (must be 2d or 3d)
        Source and target frame are equal.
    """
    if frame_from == frame_to:
        return data

    # convert timestamps to python datetimes if required
    if not isinstance(t[0], datetime.datetime):
        t = [datetime.datetime.fromtimestamp(_t, datetime.timezone.utc) for _t in t]

    if frame_cadence:
        frames = int((t[-1] - t[0]).total_seconds() // frame_cadence)
        frame_indices = [np.floor(_) for _ in np.linspace(0, frames, len(t), endpoint=False)]
        time_indices = np.linspace(0, len(t), frames, endpoint=False)

        frames = [spiceypy.pxform(frame_from, frame_to, spiceypy.datetime2et(t[int(i)]))
                  for i in time_indices]
    else:
        frames = None
        frame_indices = None

    if data.ndim == 2:
        if frame_cadence:
            for i in range(0, len(t)):
                data[i] = spiceypy.mxv(frames[int(frame_indices[i])], data[i])
        else:
            for i in range(0, len(t)):
                data[i] = spiceypy.mxv(spiceypy.pxform(frame_from, frame_to,
                                       spiceypy.datetime2et(t[i])),
                                       data[i])

        return data
    elif data.ndim == 3:
        max_workers = min(multiprocessing.cpu_count() * 2, data.shape[1])
        kernels = heliosat._spice["kernels_loaded"]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(t, data[:, i], frame_from, frame_to, frames, frame_indices, kernels)
                    for i in range(data.shape[1])]

            futures = executor.map(_worker_transform_pos, args)

        result = np.array([_ for _ in futures])

        return np.swapaxes(result, 0, 1)

    else:
        raise ValueError("data array can only be 2 or 3 dimensional")


def _worker_transform_pos(args: tuple) -> np.ndarray:
    """Worker function for transforming 3D coordinates.
    Parameters
    ----------
    args: (np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray, list)
        Function arguments as tuple.
    Returns
    -------
    np.ndarray
        Transformed 3D coordinates.
    """
    (t, data, frame_from, frame_to, frames, frame_indices, kernels) = args

    heliosat.spice.spice_init()
    heliosat.spice.spice_reload(kernels)

    if frames:
        for i in range(0, len(t)):
            data[i] = spiceypy.mxv(frames[int(frame_indices[i])], data[i])
    else:
        for i in range(0, len(t)):
            data[i] = spiceypy.mxv(spiceypy.pxform(frame_from, frame_to,
                                   spiceypy.datetime2et(t[i])),
                                   data[i])
    return data
