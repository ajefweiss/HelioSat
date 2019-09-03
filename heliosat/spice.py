# -*- coding: utf-8 -*-

"""spice.py

Implements SPICE related classes and functions.
"""

import datetime
import heliosat
import json
import logging
import numpy as np
import os
import requests
import spiceypy
import time

from heliosat.util import download_files, expand_urls


class SpiceObject(object):
    """Base class for SPICE aware objects (bodies, spacecraft etc.).
    """
    name = None
    body_name = None

    def __init__(self, name, body_name, kernel_group=None):
        """Initialize SPICE aware object and load required kernels.

        Parameters
        ----------
        name : str
            object name
        body_name : str
            SPICE object name[description]
        kernel_group : str, optional
            SPICE kernel group name, by default None
        """
        logger = logging.getLogger(__name__)

        if heliosat._spice is None:
            logger.info("running spice_init")
            spice_init()

        self.name = name
        self.body_name = body_name

        if kernel_group:
            spice_load(kernel_group)

    def trajectory(self, t, frame="J2000", observer="SUN", units="AU"):
        """Calculate body trajectory.

        Parameters
        ----------
        t : Union[list[datetime.datetime], datetime.datetime]
            observer time(s)
        frame : str, optional
            observer frame, by default "J2000"
        observer : str, optional
            observer body, by default "SUN"
        units : str, optional
            output units, by default "AU"

        Returns
        -------
        np.ndarray
            body trajectory

        Raises
        ------
        NotImplementedError
            if invalid units are given
        """
        trajectory = np.array(
            spiceypy.spkpos(
                self.body_name,
                spiceypy.datetime2et(t),
                frame,
                "NONE",
                observer
                )[0],
            dtype=np.float32
            )

        if units == "AU":
            trajectory *= 6.68459e-9
        elif units == "m":
            trajectory *= 1e3
        elif units == "km":
            pass
        else:
            raise NotImplementedError("unit \"%s\" is not supported", units)

        return trajectory


def spice_init():
    """Initialize SPICE kernels.
    """
    logger = logging.getLogger(__name__)

    kernels_path = heliosat._paths["kernels"]

    json_kernels_path = os.path.join(os.path.dirname(heliosat.__file__), "json/kernels.json")

    heliosat._spice = {"kernel_groups": None, "kernels_loaded": None}

    # json/kernels.json
    logger.info("loading available kernels")

    with open(json_kernels_path) as json_file:
        json_kernels = json.load(json_file)

    timestamp = json_kernels.get("timestamp", 0)

    # update kernels if timestamp is older than a day
    if os.path.exists(kernels_path) and timestamp > time.time() - 86400:
        logger.info("skipping check for new kernels")
    else:
        if not os.path.exists(kernels_path):
            os.makedirs(kernels_path)

        try:
            logger.info("checking for new kernels")

            json_kernels["_groups"] = dict(json_kernels["groups"])

            for group in json_kernels["_groups"]:
                json_kernels["_groups"][group] = \
                    expand_urls(json_kernels["_groups"][group])

            json_kernels["timestamp"] = time.time()

            with open(json_kernels_path, "w") as json_file:
                json.dump(json_kernels, json_file, indent=4)
        except requests.RequestException as requests_error:
            logger.error("skipping check for new kernels (%s)", requests_error)

            if "_groups" not in json_kernels:
                raise RuntimeError("expanded kernel group \"_groups\" could not be generated"
                                   "and previous version cannot be found")

    # load generic kernels
    download_files(json_kernels["generic"], kernels_path, logger=logger)

    heliosat._spice["kernel_groups"] = json_kernels["_groups"]
    heliosat._spice["kernels_loaded"] = []

    for kernel_url in json_kernels["generic"]:
        spiceypy.furnsh(os.path.join(kernels_path, kernel_url.split("/")[-1]))
        heliosat._spice["kernels_loaded"].append(kernel_url)


def spice_load(kernel_group):
    """Load SPICE kernel group.

    Parameters
    ----------
    kernel_group : str
        SPICE kernel group name
    """
    logger = logging.getLogger(__name__)

    kernels_path = heliosat._paths["kernels"]

    if kernel_group:
        kernels_required = heliosat._spice["kernel_groups"][kernel_group]

        for kernel in list(kernels_required):
            if kernel in heliosat._spice["kernels_loaded"]:
                kernels_required.remove(kernel)

        if len(kernels_required) == 0:
            return

        logger.info("loading kernel group \"%s\"", kernel_group)

        download_files(kernels_required, kernels_path, logger=logger)

        for kernel_url in kernels_required:
            kernel = os.path.join(kernels_path, kernel_url.split("/")[-1])

            logger.info("loading kernel \"%s\"", kernel)

            try:
                spiceypy.furnsh(kernel)
                heliosat._spice["kernels_loaded"].append(kernel_url)
            except spiceypy.support_types.SpiceyError:
                logger.exception("failed to load kernel \"%s\"", kernel_url)


def transform_frame(t, data, frame_from, frame_to):
    """Transform 3d vector array in between two reference frames.

    Parameters
    ----------
    t : list[datetime.datetime]
        obserer times
    data : np.ndarray
        vectory array
    frame_from : str
        source frame
    frame_to : str
        target frame

    Returns
    -------
    np.ndarray
        vector array in target frame
    """
    if frame_to and frame_from != frame_to:
        for i in range(0, len(t)):
            data[i] = spiceypy.mxv(spiceypy.pxform(frame_from, frame_to,
                                   spiceypy.datetime2et(datetime.datetime.fromtimestamp(t[i]))),
                                   data[i])
        return data
    else:
        return data
