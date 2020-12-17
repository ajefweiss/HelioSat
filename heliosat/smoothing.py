# -*- coding: utf-8 -*-

"""smoothing.py

Implements smoothing functionality.
"""

import datetime
import logging
import numba
import numpy as np

from typing import Iterable


def smooth_data(t: Iterable[datetime.datetime],
                time_raw: np.ndarray, data_raw: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
    """Smooth raw data and evaluate at timesteps t.

    Parameters
    ----------
    t : Iterable[datetime.datetime]
        Evaluation datetimes.
    time_raw : np.ndarray
        Raw time array.
    data_raw : np.ndarray
        Raw data array.

    Other Parameters
    ----------------
    smoothing: str
        Smoothing method, by default "kernel".
    smoothing_scale: float
        Smoothing scale in seconds, by default 300.

    Returns
    -------
    (np.ndarray, np.ndarray)
            Evaluation datetimes as timestamps & smoothed data array.

    Raises
    ------
    NotImplementedError
        If smoothing method is not implemented.
    """
    logger = logging.getLogger(__name__)

    time_smooth = np.array([_t.timestamp() for _t in t])
    data_smooth = np.zeros((len(t), data_raw.shape[1]))

    smoothing = kwargs.get("smoothing", "kernel")
    smoothing_scale = kwargs.get("smoothing_scale", 300)

    if smoothing in ["average", "moving_average", "mean"]:
        _smoothing_mean(time_smooth, time_raw, data_raw, data_smooth, smoothing_scale)
    elif smoothing in ["kernel", "kernel_gaussian", "gaussian"]:
        _smoothing_gaussian_kernel(time_smooth, time_raw, data_raw, data_smooth, smoothing_scale)
    elif smoothing in ["linear", "linear_interpolation"]:
        data_smooth = np.array([np.interp(time_smooth, time_raw, data_raw[:, i])
                                for i in range(data_raw.shape[1])])
        return time_smooth, data_smooth.T
    elif smoothing in ["closest"]:
        return _smoothing_closest(time_smooth, time_raw, data_raw, data_smooth)
    else:
        logger.exception("smoothing method \"%s\" is not implemented", kwargs.get("smoothing"))
        raise NotImplementedError("smoothing method \"%s\" is not implemented",
                                  kwargs.get("smoothing"))

    return time_smooth, data_smooth


@numba.njit(parallel=False)
def _smoothing_closest(t, time_raw, data_raw, data_smooth):
    """Smooth data using closest.

    Parameters
    ----------
    t : list[float]
        Evaluation times as timestamps.
    time_raw : np.ndarray
        Raw time array.
    data_raw : np.ndarray
        Raw data array.
    data_smooth : np.ndarray
        Smoothed data array.
    smoothing_scale : float
        Smoothing scale in seconds.
    """
    t_actual = np.zeros_like(t)

    for i in numba.prange(len(t)):
        index = np.argmin(np.abs(t[i] - time_raw))

        t_actual[i] = time_raw[index]
        data_smooth[i] = data_raw[index]

    return t_actual, data_smooth


@numba.njit(parallel=True)
def _smoothing_mean(t, time_raw, data_raw, data_smooth, smoothing_scale):
    """Smooth data using moving average.

    Parameters
    ----------
    t : list[float]
        Evaluation times as timestamps.
    time_raw : np.ndarray
        Raw time array.
    data_raw : np.ndarray
        Raw data array.
    data_smooth : np.ndarray
        Smoothed data array.
    smoothing_scale : float
        Smoothing scale in seconds.
    """
    for i in numba.prange(len(t)):
        total = 0
        dims = data_raw.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_raw)):
            if np.abs(time_raw[j] - t[i]) < smoothing_scale:
                total += 1

                for k in range(0, len(vector)):
                    vector[k] += data_raw[j, k]

        for k in range(0, len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total


@numba.njit(parallel=True)
def _smoothing_gaussian_kernel(t, time_raw, data_raw, data_smooth, smoothing_scale):
    """Smooth data using a gaussian kernel.

    Parameters
    ----------
    t : list[float]
        Evaluation times as timestamps.
    time_raw : np.ndarray
        Raw time array.
    data_raw : np.ndarray
        Raw data array.
    data_smooth : np.ndarray
        Smoothed data array.
    smoothing_scale : float
        Smoothing scale in seconds.
    """
    for i in numba.prange(len(t)):
        total = 0
        dims = data_raw.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_raw)):
            if np.abs(time_raw[j] - t[i]) < 3 * smoothing_scale and not np.isnan(data_raw[j, 0]):
                kernel = np.exp(-(time_raw[j] - t[i]) ** 2 / 2 / smoothing_scale ** 2)

                total += kernel

                for k in range(0, len(vector)):
                    vector[k] += kernel * data_raw[j, k]

        for k in range(0, len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total
