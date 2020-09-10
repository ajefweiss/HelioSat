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


def transform_lonlat(t, lons, lats, frame_from, frame_to):
    """Transform longitude/latitude direction from one reference frame to another.

    Parameters
    ----------
    t : datetime.datetime
        Evaluation datetimes.
    lons : np.ndarray
        Longitudes array.
    lats : np.ndarray
        Latitudes array.
    """
    pass


def transform_frame(t, data, frame_from, frame_to, frame_cadence=None):
    """Transform 3D coordinates from one reference frame to another.

    Parameters
    ----------
    t : list[datetime.datetime]
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
        If data array has the wrong dimensions (must be 2d or 3d).
    """
    if not frame_to or frame_from == frame_to:
        raise ValueError("no target frame defined (or source frame and target frame are equal)")

    # convert timestamps to python datetimes if required
    if not isinstance(t[0], datetime.datetime):
        t = [datetime.datetime.fromtimestamp(_t) for _t in t]

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

            futures = executor.map(worker_transform_frame, args)

        result = np.array([_ for _ in futures])

        return np.swapaxes(result, 0, 1)

    else:
        raise ValueError("data array can only be 2 or 3 dimensional")


def worker_transform_frame(args):
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

    if len(heliosat._spice["kernels_loaded"]) == 0:
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
