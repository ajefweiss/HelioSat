# -*- coding: utf-8 -*-

"""routines.py

Implements data routines.
"""

import datetime as dt
import logging as lg
from typing import Sequence, Tuple, Union

import numpy as np
import spiceypy
from scipy.signal import detrend, welch


# TODO: replace with lombscargle
def power_spectral_density(
    dtp: Sequence[dt.datetime],
    dk: np.ndarray,
    format_for_fft: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean power spectrum distribution from a magnetic field measurements.
    If multiple measurements are given the result is averaged over all.
    """
    n_s = int(((dtp[-1] - dtp[0]).total_seconds() / 3600) - 1)

    # compute sample frequency
    dtp_ts = np.array([_.timestamp() for _ in dtp])
    dtp_diff = dtp_ts[1:] - dtp_ts[:-1]

    deviation = np.std(dtp_diff) / np.median(dtp_diff)

    if not np.all(dtp_diff == dtp_diff[0]) and deviation >= 0.01:
        raise ValueError(
            "datetimes are not equidistant from each other (significant deviations)"
        )
    elif deviation < 0.01:
        lg.warning(
            "datetimes are not equidistant from each other, ignoring due to low amount of outliers"
        )

    sampling_freq = 1 / np.median(dtp_diff)

    if dk.ndim == 1:
        p_bA = detrend(dk[:], type="linear", bp=n_s)

        wF, wS = welch(p_bA, fs=sampling_freq, nperseg=None)
    else:
        p_bA = [detrend(dk[:, i], type="linear", bp=n_s) for i in range(dk.shape[1])]

        results = [
            welch(p_bA[i], fs=sampling_freq, nperseg=None) for i in range(dk.shape[1])
        ]

        wSs = np.array([_[1] for _ in results])

        wF = results[0][0]
        wS = np.mean(wSs, axis=0) / dk.shape[1]

    if format_for_fft:
        # convert into suitable form for fft
        fF = np.fft.fftfreq(len(wS), d=sampling_freq)
        fS = np.zeros((len(fF)))

        for i in range(len(fF)):
            k = np.abs(fF[i])
            fS[i] = np.sqrt(wS[np.argmin(np.abs(k - wF))])

        return fF, fS
    else:
        return wF, wS


def transform_reference_frame(
    dtp: Union[dt.datetime, Sequence[dt.datetime]],
    vec_array: np.ndarray,
    reference_frame_from: str,
    reference_frame_to: str,
) -> np.ndarray:
    if reference_frame_from == reference_frame_to:
        return vec_array

    if isinstance(dtp, dt.datetime):
        dtp = [dtp] * len(vec_array)

    # convert to datetime objects
    if not isinstance(dtp[0], dt.datetime):
        dtp = [dt.datetime.fromtimestamp(_t, dt.timezone.utc) for _t in dtp]

    vec_array_new = np.zeros_like(vec_array)

    if vec_array.ndim == 1:
        vec_array.reshape(-1, len(vec_array))

    if vec_array.ndim == 2:
        for i in range(0, len(dtp)):
            vec_array_new[i] = spiceypy.mxv(
                spiceypy.pxform(
                    reference_frame_from,
                    reference_frame_to,
                    spiceypy.datetime2et(dtp[i]),
                ),
                vec_array[i],
            )

    return vec_array_new
