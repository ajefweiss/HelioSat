# -*- coding: utf-8 -*-

import datetime
import heliosat
import json
import logging
import numpy as np
import os
import re
import requests
import requests_ftp
import spiceypy
import time

from bs4 import BeautifulSoup

from heliosat.util import download_files


class SpiceObject(object):
    """Implemenmts functionality for SPICE aware objects.
    """
    name = None
    bodyname = None

    def __init__(self, name, bodyname, kernel_group=None):
        """Initialize SPICE aware object and load required kernels.

        Parameters
        ----------
        name : str
            object/spacecraft name
        bodyname : str
            SPICE body name
        kernel_group : str, optional
            kernel group name, by default None
        """
        logger = logging.getLogger(__name__)

        home_path = os.path.join(os.path.expanduser("~"), ".heliosat")
        mod_path = os.path.dirname(heliosat.__file__)
        kernels_path = os.path.join(home_path, "kernels")

        self.name = name
        self.bodyname = bodyname

        if heliosat._spacecraft_available is None:
            # populate available spacecraft
            logger.info("loading available spacecraft")

            with open(os.path.join(mod_path, "json/spacecraft.json")) as json_file:
                heliosat._spacecraft_available = json.load(json_file)

        if heliosat._kernels_available is None:
            # populate available kernels
            logger.info("loading available kernels")

            json_path = os.path.join(mod_path, "json/kernels.json")

            with open(json_path) as json_file:
                json_kernels = json.load(json_file)

            timestamp = json_kernels.get("timestamp", 0)

            if os.path.exists(kernels_path) and timestamp > time.time() - 24 * 3600:
                logger.info("skipping the check for new kernels")
            else:
                if not os.path.exists(kernels_path):
                    os.makedirs(kernels_path)

                try:
                    logger.info("checking for new kernels")

                    json_kernels["groups_expanded"] = dict(json_kernels["groups"])

                    for group in json_kernels["groups_expanded"]:
                        json_kernels["groups_expanded"][group] = \
                            expand_urls(json_kernels["groups_expanded"][group])

                    json_kernels["timestamp"] = time.time()

                    with open(json_path, "w") as json_file:
                        json.dump(json_kernels, json_file, indent=4)
                except requests.HTTPError as http_error:
                    logger.error("skipping the check for new kernels ({0})".format(http_error))

                    if "groups_expanded" not in json_kernels:
                        raise RuntimeError("expanded kernel group definitions not found")

            # load generic kernels
            download_files(json_kernels["generic"], kernels_path, logger=logger)

            for kernel in json_kernels["generic"]:
                kernel_path = os.path.join(kernels_path, kernel.split("/")[-1])

                logger.info("loading kernel \"{0}\"".format(kernel_path))

                spiceypy.furnsh(kernel_path)

            heliosat._kernels_available = json_kernels["groups_expanded"]
            heliosat._kernels_loaded = []

        # load spacecraft
        if name and name in heliosat._spacecraft_available:
            def iter_attr(child, varname):
                if isinstance(child, dict):
                    for key in child:
                        iter_attr(child[key], varname="{0}_{1}".format(varname, key))
                else:
                    try:
                        setattr(self, varname,
                                datetime.datetime.strptime(child, "%Y-%m-%dT%H:%M:%S.%f"))
                    except (TypeError, ValueError):
                        setattr(self, varname, child)

            iter_attr(heliosat._spacecraft_available[name], "")

            if kernel_group is None and hasattr(self, "_kernel_group"):
                kernel_group = getattr(self, "_kernel_group")

            # set and create data folder
            setattr(self, "_data_folder", os.path.join(home_path, "data", self.name))

            if not os.path.exists(getattr(self, "_data_folder")):
                os.makedirs(getattr(self, "_data_folder"))

        # load kernel_group
        if kernel_group:
            kernels_required = heliosat._kernels_available[kernel_group]

            for kernel in list(kernels_required):
                if kernel in heliosat._kernels_loaded:
                    kernels_required.remove(kernel)

            if len(kernels_required) == 0:
                return

            logger.info("loading kernel group \"{}\"".format(kernel_group))

            download_files(kernels_required, kernels_path, logger=logger)

            for kernel in kernels_required:
                kernel_path = os.path.join(kernels_path, kernel.split("/")[-1])

                logger.info("loading kernel \"{0}\"".format(kernel_path))

                heliosat._kernels_loaded.append(kernel)

                spiceypy.furnsh(kernel_path)

    def trajectory(self, t, reference_frame="J2000", observer="SUN", units="AU"):
        """Calculate body trajectory at observer time t.

        Parameters
        ----------
        t : Iterable[datetime.datetime]
            observer time
        reference_frame : str, optional
            observer reference frame, by default "J2000"
        observer : str, optional
            observer body name, by default "SUN"
        units : str, optional
            output units, by default "AU"

        Returns
        -------
        np.ndarray
            body trajectory at observer times
        """
        trajectory = np.array(spiceypy.spkpos(self.bodyname, spiceypy.datetime2et(t),
                                              reference_frame, "NONE", observer)[0])

        if units == "AU":
            trajectory *= 6.68459e-9
        elif units == "m":
            trajectory *= 1e3
        elif units == "km":
            pass
        else:
            raise NotImplementedError("unit \"{0}\" is not supported".format(units))

        return trajectory


def expand_urls(urls):
    """Expand urls (only those prefixed with a "$" character) by searching the parent html page or
    ftp file list for regex matches.

    Parameters
    ----------
    urls : list
        list of urls, normal or ones with regex patterns (prefixed with a $)

    Returns
    -------
    list
        list of expanded urls
    """
    urls_all = [_ for _ in urls if _[0] != "$"]

    for url in [_ for _ in urls if _[0] == "$"]:
        if url[1:].startswith("http"):
            url_parent = "/".join(url[1:].split("/")[:-1])
            url_regex = url.split("/")[-1]

            response = requests.get(url_parent)

            if response.ok:
                response_text = response.text
            else:
                return response.raise_for_status()

            # match all urls"s with regex pattern
            soup = BeautifulSoup(response_text, "html.parser")

            for url_child in [_.get("href") for _ in soup.find_all("a")]:
                if url_child and re.match(url_regex, url_child):
                    urls_all.append("/".join([url_parent, url_child]))
        elif url[1:].startswith("ftp"):
            url_parent = "/".join(url[1:].split("/")[:-1])
            url_regex = url.split("/")[-1]

            response = requests_ftp.ftp.FTPSession().list(url_parent)

            if response.ok:
                response_text = response.text
            else:
                return response.raise_for_status()

            # match all urls"s with regex pattern
            filenames = [line.split()[-1] for line in response.content.decode("utf-8").splitlines()]

            for filename in filenames:
                if re.match(url_regex, filename):
                    urls_all.append("/".join([url_parent, filename]))

    return urls_all
