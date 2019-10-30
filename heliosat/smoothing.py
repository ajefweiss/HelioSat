# -*- coding: utf-8 -*-

"""smoothing.py

Implements smoothing functions.
"""

import logging
import numba
import numpy as np


def smooth_data(t, time_raw, data_raw, **kwargs):
    """Smooth raw data and evaluate at timesteps t.

    Parameters
    ----------
    t : list[datetime.datetime]
        evaluation times
    time_raw : np.ndarray
        raw time array
    data_raw : np.ndarray
        raw data array

    Returns
    -------
    (np.ndarray, np.ndarray)
        smoothed time and data array

    Raises
    ------
    NotImplementedError
        if smoothing method is not implemented
    """
    logger = logging.getLogger(__name__)

    time_smooth = np.array([_t.timestamp() for _t in t])
    data_smooth = np.zeros((len(t), data_raw.shape[1]), dtype=np.float32)

    if kwargs.get("smoothing") == "kernel" or kwargs.get("smoothing") == "kernel_gaussian":
        smoothing_scale = kwargs.get("smoothing_scale", 300)

        kernel_smoothing_gaussian(time_smooth, time_raw, data_raw, data_smooth, smoothing_scale)
    elif kwargs.get("smoothing") == "spline":
        raise NotImplementedError
    else:
        logger.exception("smoothing method \"%s\" is not implemented", kwargs.get("smoothing"))
        raise NotImplementedError("smoothing method \"%s\" is not implemented",
                                  kwargs.get("smoothing"))

    # remove NaN's
    nan_mask = np.invert(np.isnan(data_smooth[:, 0]))
    time_smooth = time_smooth[nan_mask]
    data_smooth = data_smooth[nan_mask]

    return time_smooth, data_smooth


@numba.njit("void(f8[:], f8[:], f4[:, :], f4[:, :], f8)", parallel=True)
def kernel_smoothing_gaussian(t, time_raw, data_raw, data_smooth, smoothing_scale):
    """Smooth data using a gaussian kernel.

    Parameters
    ----------
    t : list[float]
        evaluation times (timestamp)
    time_raw : np.ndarray
        raw time array
    data_raw : np.ndarray
        raw data array
    data_smooth : np.ndarray
        smoothed data array (output)
    smoothing_scale : float
        smoothing scale (in seconds)
    """
    for i in numba.prange(len(t)):
        total = 0
        dims = data_raw.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_raw)):
            if np.abs(time_raw[j] - t[i]) < 2 * smoothing_scale:
                kernel = np.exp(-(time_raw[j] - t[i]) ** 2 / 2 / smoothing_scale ** 2)

                total += kernel

                for k in range(0, len(vector)):
                    vector[k] += kernel * data_raw[j, k]

        for k in range(0,  len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total
