# -*- coding: utf-8 -*-

"""routines.py

Implements data routines.
"""

import datetime as dt
from typing import Optional, Sequence, Union

import numpy as np
import spiceypy


def transform_reference_frame(
    dtp: Union[dt.datetime, Sequence[dt.datetime]],
    vec_array: np.ndarray,
    reference_frame_from: str,
    reference_frame_to: str,
    batch_size: Optional[int] = None,
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
            if batch_size and batch_size > 1 and isinstance(batch_size, int):
                if i % batch_size == 0:
                    pxform = spiceypy.pxform(
                        reference_frame_from,
                        reference_frame_to,
                        spiceypy.datetime2et(dtp[i]),
                    )

                vec_array_new[i] = spiceypy.mxv(pxform, vec_array[i])
            else:
                pxform = spiceypy.pxform(
                    reference_frame_from,
                    reference_frame_to,
                    spiceypy.datetime2et(dtp[i]),
                )
                vec_array_new[i] = spiceypy.mxv(pxform, vec_array[i])

    return vec_array_new
