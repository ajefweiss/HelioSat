# -*- coding: utf-8 -*-

"""coordinates.py

Implements coordinate transformation functions requiring SPICE.
"""

import datetime
import heliosat
import numpy as np
import spiceypy

from typing import Sequence, Union


def transform_reference_frame(dt: Union[datetime.datetime, Sequence[datetime.datetime]], vec_array: np.ndarray, reference_frame_from: str, reference_frame_to: str) -> np.ndarray:
    if reference_frame_from == reference_frame_to:
        return vec_array

    if isinstance(dt, datetime.datetime):
        dt = [dt] * len(vec_array)
    
    # convert to datetimeobjects
    if not isinstance(dt[0], datetime.datetime):
        dt = [datetime.datetime.fromtimestamp(_t, datetime.timezone.utc) for _t in dt]

    vec_array_new = np.zeros_like(vec_array)

    if vec_array.ndim == 1:
        vec_array.reshape(-1, len(vec_array))

    if vec_array.ndim == 2:
        for i in range(0, len(dt)):
            vec_array_new[i] = spiceypy.mxv(spiceypy.pxform(reference_frame_from, reference_frame_to, spiceypy.datetime2et(dt[i])), vec_array[i])

    return vec_array_new
