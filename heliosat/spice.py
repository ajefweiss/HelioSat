# -*- coding: utf-8 -*-

"""spice.py

Implements the SPICE base class and SPICE related functions.
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

from heliosat.download import download_files, urls_expand
from typing import Iterable, List, Optional, Union


class SpiceObject(object):
    """Base SPICE class for SPICE aware objects (bodies, spacecraft etc.).
    """
    def __init__(self, name: str, body_name: str, kernel_group: Optional[str] = None,
                 skip_init: bool = False, force_download: bool = False):
        """Initialize SPICE aware object and load required kernels if given.

        Parameters
        ----------
        name : str
            Object name (for spacecrafts as defined in the spacecraft.json file).
        body_name : str
            SPICE object name.
        kernel_group : str, optional
            SPICE kernel group name, by default None.
        skip_init: bool, optional
            Initialize default spice kernels, by default True.
        force_download: bool, optional
            Force kernel downloads (otherwise skip remote checks), by default False.
        """
        logger = logging.getLogger(__name__)

        if skip_init is False and heliosat._spice is None:
            logger.info("running spice_init")
            spice_init(force_download)

        self.name = name
        self.body_name = body_name

        if kernel_group:
            spice_load(kernel_group, force_download)

    def trajectory(self, t: Union[datetime.datetime, Iterable[datetime.datetime]],
                   frame: str = "J2000", observer: str = "SUN", units: str = "AU"):
        """Evaluate body trajectory at given datetimes.

        Parameters
        ----------
        t : Union[datetime.datetime, Iterable[datetime.datetime]]
            Evaluate datetime(s).
        frame : str, optional
            Trajectory reference frame, by default "J2000".
        observer : str, optional
            Observer body name, by default "SUN".
        units : str, optional
            Output units, by default "AU".

        Returns
        -------
        np.ndarray
            Body trajectory.

        Raises
        ------
        NotImplementedError
            If units are invalid.
        """
        logger = logging.getLogger(__name__)

        if heliosat._spice is None:
            logger.info("running spice_init")
            spice_init(False)

        trajectory = np.array(
            spiceypy.spkpos(
                self.body_name,
                spiceypy.datetime2et(t),
                frame,
                "NONE",
                observer
            )[0]
        )

        if units == "AU":
            trajectory *= 6.68459e-9
        elif units == "m":
            trajectory *= 1e3
        elif units == "km":
            pass
        else:
            logger.exception("unit \"%s\" is not supported", units)
            raise NotImplementedError("unit \"%s\" is not supported", units)

        return trajectory


def spice_init(force_download: bool = False):
    """Initialize SPICE kernels.

    Parameters
    ----------
    force_download: bool, optional
            Force kernel downloads (otherwise skip remote checks), by default False.
    """
    logger = logging.getLogger(__name__)

    # clear all
    spiceypy.kclear()

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
                    urls_expand(json_kernels["_groups"][group])

            json_kernels["timestamp"] = time.time()

            with open(json_kernels_path, "w") as json_file:
                json.dump(json_kernels, json_file, indent=4)
        except requests.RequestException as requests_error:
            logger.error("skipping check for new kernels (%s)", requests_error)

            if "_groups" not in json_kernels:
                logger.exception("expanded kernel group \"_groups\" could not be generated"
                                 "and previous version cannot be found")
                raise RuntimeError("expanded kernel group \"_groups\" could not be generated"
                                   "and previous version cannot be found")

    # load generic kernels
    download_files(json_kernels["generic"], kernels_path, force=force_download, logger=logger)

    heliosat._spice["kernel_groups"] = json_kernels["_groups"]
    heliosat._spice["kernels_loaded"] = []

    for kernel_url in json_kernels["generic"]:
        spiceypy.furnsh(os.path.join(kernels_path, kernel_url.split("/")[-1]))
        heliosat._spice["kernels_loaded"].append(kernel_url)


def spice_load(kernel_group: str, force_download: bool = False):
    """Load SPICE kernel group.

    Parameters
    ----------
    kernel_group : str
        SPICE kernel group name as defined in the spacecraft.json file.
    force_download: bool, optional
            Force kernel downloads (otherwise skip remote checks), by default False.
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

        logger.debug("loading kernel group \"%s\"", kernel_group)

        download_files(kernels_required, kernels_path, force=force_download, logger=logger)

        for kernel_url in kernels_required:
            kernel = os.path.join(kernels_path, kernel_url.split("/")[-1])

            logger.debug("loading kernel \"%s\"", kernel)

            try:
                spiceypy.furnsh(kernel)
                heliosat._spice["kernels_loaded"].append(kernel_url)
            except spiceypy.support_types.SpiceyError:
                logger.exception("failed to load kernel \"%s\"", kernel_url)


def spice_reload(kernel_urls: List[str], force_download: bool = False):
    """Load SPICE kernels by url. This function is meant to be used in child processes that need
    to reload all SPICE kernels.

    Parameters
    ----------
    kernel_urls : List[str]
        SPICE kernel urls.
    force_download: bool, optional
            Force kernel downloads (otherwise skip remote checks), by default False.
    """
    logger = logging.getLogger(__name__)

    kernels_path = heliosat._paths["kernels"]

    kernels_required = kernel_urls
    
    for kernel in list(kernels_required):
        if kernel in heliosat._spice["kernels_loaded"]:
            kernels_required.remove(kernel)

    if len(kernels_required) == 0:
        return

    download_files(kernels_required, kernels_path, force=force_download, logger=logger)

    for kernel_url in kernels_required:
        kernel = os.path.join(kernels_path, kernel_url.split("/")[-1])

        logger.debug("loading kernel \"%s\"", kernel)
        try:
            spiceypy.furnsh(kernel)
            heliosat._spice["kernels_loaded"].append(kernel_url)
        except spiceypy.support_types.SpiceyError:
            logger.exception("failed to load kernel \"%s\"", kernel_url)
