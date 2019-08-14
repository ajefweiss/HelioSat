# -*- coding: utf-8 -*-

import logging
import numba
import numpy as np
import time


def smooth_gaussian_1d(t, time_data, raw_data, smoothing_scale=120):
    """Smooth data set using a gaussian kernel and evaluate the result at times t.

    Parameters
    ----------
    t : Iterable[datetime.datetime]
        observer time
    time_data : np.array
        time data array (in seconds)
    raw_data : np.array
        raw data array
    smoothing_scale : int, optional
        kernel radius in seconds, by default 120
    """
    logger = logging.getLogger(__name__)

    smooth_data = np.zeros((len(t), 3), dtype=np.float32)

    timer = time.time()

    smoothing_adaptive_gaussian(np.array([_t.timestamp() for _t in t]), time_data, raw_data,
                                smooth_data, float(smoothing_scale))
    logger.info("smoothing with adaptive gaussian method, scale={0}s "
                "({1:.1f}s)".format(smoothing_scale, time.time() - timer))

    return smooth_data


@numba.njit("f8(f8, f8, f8)", cache=True)
def _gaussian_kernel(x_0, x_i, kernel_radius):
    """Gaussian kernel
    """
    return np.exp(-(x_0 - x_i) ** 2 / 2 / kernel_radius ** 2)


@numba.njit("void(f8[:], f8[:], f4[:, :], f4[:, :], f8)", cache=True, parallel=True)
def smoothing_adaptive_gaussian(eval_x, data_x, data_y, eval_y, smoothing_scale):
    """Smooth multidimensional data set with a adaptive gaussian kernel. The gaussian kernel radius
    is lowered when the data appears highly variable.

    Parameters
    ----------
    eval_x : np.ndarray
        The x-coordinates at which to evaluate the smoothed values.
    data_x : np.ndarray
        The x-coordinates of the data points.
    data_y : np.ndarray
        The y-coordinates of the data points.
    eval_y : np.ndarray
        The y-coordinates which are to be evaluated (output array).
    smoothing_scale : float
        smoothing scale (seconds)
    """
    # evaluate typical variance in the data set
    var = np.zeros((len(eval_x),))

    for i in numba.prange(0, len(eval_x)):
        indices = (data_x > eval_x[i] - 2.5 * smoothing_scale) & \
            (data_x < eval_x[i] + 2.5 * smoothing_scale)
        var[i] = np.std(data_y[indices])

    var_max = np.max(var)
    var_min = np.min(var)
    var_diff = var_max - var_min
    var_rat = var_min / var_max

    # calculate smoothing scales for each window
    scales = np.zeros((len(eval_x),))

    for i in numba.prange(0, len(eval_x)):
        scales[i] = (var_rat - 1) / var_diff * var[i] + 1 - (var_rat - 1) / var_diff * var_min

        if scales[i] > np.median(var):
            scales[i] = 1

    for i in numba.prange(0, len(eval_x)):
        total = 0
        dims = data_y.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_x)):
            if np.abs(data_x[j] - eval_x[i]) < 2.5 * scales[i] * smoothing_scale:
                kernel = _gaussian_kernel(data_x[j], eval_x[i], scales[i] * smoothing_scale)

                total += kernel

                for k in range(0, len(vector)):
                    vector[k] += kernel * data_y[j, k]

        for k in range(0,  len(vector)):
            if total == 0:
                eval_y[i, k] = 0
            else:
                eval_y[i, k] = vector[k] / total
