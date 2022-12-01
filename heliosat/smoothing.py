# -*- coding: utf-8 -*-

"""smoothing.py

Implements simple smoothing functions. Designed for internal use only.
"""

import datetime as dt
import logging as lg
import numpy as np

from typing import Any, Sequence, Tuple

# import numba if available, otherwise define custom decorator
try:
    import numba as nb
except:
    class nb(object):
        def njit(fn):
            lg.info("function %s: numba package not installed, function may be slow", fn)
            return fn


def smooth_data(dt: Sequence[dt.datetime], dt_r: np.ndarray, dk_r: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    time_smooth = np.array([_t.timestamp() for _t in dt])
    data_smooth = np.zeros((len(dt), dk_r.shape[1]))

    smoothing = kwargs.get("smoothing", "closest")
    smoothing_scale = kwargs.get("smoothing_scale", 300)

    if smoothing in ["average", "moving_average", "mean"]:
        _smoothing_mean(time_smooth, dt_r, dk_r, data_smooth, smoothing_scale)
    elif smoothing in ["kernel", "kernel_gaussian", "gaussian"]:
        _smoothing_gaussian_kernel(time_smooth, dt_r, dk_r, data_smooth, smoothing_scale)
    elif smoothing in ["linear", "linear_interpolation"]:
        data_smooth = np.array([np.interp(time_smooth, dt_r, dk_r[:, i])
                                for i in range(dk_r.shape[1])])
        data_smooth = data_smooth.T
    elif smoothing in ["closest"]:
        time_smooth, data_smooth = _smoothing_closest(time_smooth, dt_r, dk_r, data_smooth)
    else:
        raise NotImplementedError("smoothing method \"{0!s}\" is not implemented".format(kwargs.get("smoothing")))

    return time_smooth, data_smooth


@nb.njit
def _smoothing_closest(dt: np.ndarray, dt_r: np.ndarray, dk_r: np.ndarray, data_smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t_actual = np.zeros_like(dt)

    for i in range(len(dt)):
        index = np.argmin(np.abs(dt[i] - dt_r))

        t_actual[i] = dt_r[index]
        data_smooth[i] = dk_r[index]

    return t_actual, data_smooth


@nb.njit
def _smoothing_mean(dt: np.ndarray, dt_r: np.ndarray, dk_r: np.ndarray, data_smooth: np.ndarray, smoothing_scale: np.ndarray) -> None:
    for i in range(len(dt)):
        total = 0
        dims = dk_r.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(dk_r)):
            if np.abs(dt_r[j] - dt[i]) < smoothing_scale:
                total += 1

                for k in range(0, len(vector)):
                    vector[k] += dk_r[j, k]

        for k in range(0, len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total


@nb.njit
def _smoothing_gaussian_kernel(dt: np.ndarray, dt_r: np.ndarray, dk_r: np.ndarray, data_smooth: np.ndarray, smoothing_scale: np.ndarray) -> None:
    for i in range(len(dt)):
        total = 0
        dims = dk_r.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(dk_r)):
            if np.abs(dt_r[j] - dt[i]) < 3 * smoothing_scale and not np.isnan(dk_r[j, 0]):
                kernel = np.exp(-(dt_r[j] - dt[i]) ** 2 / 2 / smoothing_scale ** 2)

                total += kernel

                for k in range(0, len(vector)):
                    vector[k] += kernel * dk_r[j, k]

        for k in range(0, len(vector)):
            if total == 0:
                data_smooth[i, k] = np.nan
            else:
                data_smooth[i, k] = vector[k] / total
