# -*- coding: utf-8 -*-

"""smoothing.py

Implements smoothing functions.
"""

import numba
import numpy as np


def smooth_data(t, time_raw, data_raw, **kwargs):
    """Smooth raw data and evaluate at times t.

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
    time_smooth = np.array([_t.timestamp() for _t in t])
    data_smooth = np.zeros((len(t), data_raw.shape[1]), dtype=np.float32)

    if kwargs.get("smoothing") == "kernel":
        smoothing_scale = kwargs.get("smoothing-scale", 300)

        kernel_smoothing(time_smooth, time_raw, data_raw, data_smooth, smoothing_scale)
    else:
        raise NotImplementedError("smoothing method \"%s\" is not implemented",
                                  kwargs.get("smoothing"))

    return time_smooth, data_smooth


@numba.njit("f8(f8, f8, f8)", cache=True)
def kernel_smoothing_gaussian_kernel(x_0, x_i, r):
    """Gaussian kernel.
    """
    return np.exp(-(x_0 - x_i) ** 2 / 2 / r ** 2)


@numba.njit("void(f8[:], f8[:], f4[:, :], f4[:, :], f8)", cache=True, parallel=True)
def kernel_smoothing(t, time_raw, data_raw, data_smooth, smoothing_scale):
    """Smooth data using a gaussian kernel.

    Parameters
    ----------
    t : list[datetime.datetime]
        evaluation times
    time_raw : np.ndarray
        raw time array
    data_raw : np.ndarray
        raw data array
    data_smooth : np.ndarray
        smoothed data array (output)
    smoothing_scale : np.ndarray
        smoothing scale (in seconds)
    """
    for i in numba.prange(0, len(t)):
        total = 0
        dims = data_raw.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_raw)):
            if np.abs(time_raw[j] - t[i]) < 2 * smoothing_scale:
                kernel = kernel_smoothing_gaussian_kernel(time_raw[j], t[i], smoothing_scale)

                total += kernel

                for k in range(0, len(vector)):
                    vector[k] += kernel * data_raw[j, k]

        for k in range(0,  len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total
