# -*- coding: utf-8 -*-

"""transform.py

Implements coordinate transformation functions requiring SPICE kernels.
"""

import datetime as dt
import numpy as np
import spiceypy

from typing import Sequence, Union


def transform_reference_frame(dtp: Union[dt.datetime, Sequence[dt.datetime]], vec_array: np.ndarray, reference_frame_from: str, reference_frame_to: str) -> np.ndarray:
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
            vec_array_new[i] = spiceypy.mxv(spiceypy.pxform(reference_frame_from, reference_frame_to, spiceypy.datetime2et(dtp[i])), vec_array[i])

    return vec_array_new
