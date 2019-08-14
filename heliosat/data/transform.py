# -*- coding: utf-8 -*-

import spiceypy


def transform_ref_frame(t, arr, reference_frame_from, reference_frame_to):
    """Transform vectors from one reference frame to another
    Parameters
    ----------
    t : Iterable[datetime.datetime]
        observer times
    arr : np.ndarray
        vector array in first reference frame
    reference_frame_from : str
        first reference frame
    reference_frame_to : str
        second reference frame
    Returns
    -------
    np.ndarray
        vector array in second reference frame
    """
    if reference_frame_to and reference_frame_from != reference_frame_to:
        for i in range(0, len(t)):
            arr[i] = spiceypy.mxv(spiceypy.pxform(reference_frame_from, reference_frame_to,
                                  spiceypy.datetime2et(t[i])), arr[i])

        return arr
    else:
        return arr
