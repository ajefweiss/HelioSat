# -*- coding: utf-8 -*-

import numba
import numpy as np


@numba.njit("f8(f8, f8, f8)", cache=True)
def _gaussian_kernel(x_0, x_i, kernel_radius):
    """Gaussian kernel
    """
    return np.exp(-(x_0 - x_i) ** 2 / 2 / kernel_radius ** 2)


@numba.njit("void(f8[:], f8[:], f4[:, :], f4[:, :], f8)", cache=True, parallel=True)
def smoothing_average_kernel(eval_x, data_x, data_y, eval_y, smoothing_scale):
    """Smooth multidimensional data set with a average kernel.

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
    for i in numba.prange(0, len(eval_x)):
        total = 0
        dims = data_y.shape[1]
        vector = np.zeros((dims,))

        for j in range(0, len(data_x)):
            if np.abs(data_x[j] - eval_x[i]) < 2.5 * smoothing_scale:
                total += 1

                for k in range(0, len(vector)):
                    vector[k] += data_y[j, k]

        for k in range(0,  len(vector)):
            if total == 0:
                eval_y[i, k] = 0
            else:
                eval_y[i, k] = vector[k] / total


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


@numba.njit("void(f8[:], f8[:], f4[:, :], f4[:, :], f8)", cache=True)
def smoothing_adaptive_gaussian_normalized(eval_x, data_x, data_y, eval_y, smoothing_scale):
    """Smooth multidimensional data set with a adaptive gaussian kernel and fixed vector length.

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
    lengths = np.empty((len(data_x), 1), dtype=np.float32)
    lengths_smooth = np.empty((len(eval_x), 1), dtype=np.float32)

    for i in range(0, len(data_x)):
        lengths[i] = np.sqrt(data_y[i, 0] ** 2 + data_y[i, 1] ** 2 + data_y[i, 2] ** 2)

    smoothing_adaptive_gaussian(eval_x, data_x, lengths, lengths_smooth,
                                smoothing_scale)

    smoothing_adaptive_gaussian(eval_x, data_x, data_y, eval_y, smoothing_scale)

    for i in range(0, len(eval_x)):
        eval_y[i] *= lengths_smooth[i] / np.linalg.norm(eval_y[i])
