# -*- coding: utf-8 -*-

"""spice.py
"""

import concurrent.futures
import logging as lg
import os
from runpy import run_path
from typing import List, Union

import spiceypy

import heliosat

from .spacecraft import Spacecraft
from .util import fetch_url, load_json, url_basename, url_dirname, url_regex_files, url_regex_resolve


class SpiceKernel(object):
    """SPICE Kernel class."""

    url: str
    loaded: bool

    def __init__(self, url: str, data_path: str) -> None:
        self.url = url
        self.file_name = url_basename(url)
        self.file_path = os.path.join(data_path, "kernels", self.file_name)

    @property
    def is_available(self) -> bool:
        return os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0

    def prepare(self, force_download: bool = False) -> None:
        logger = lg.getLogger(__name__)

        # check local availability
        if self.is_available and not force_download:
            return

        try:
            file_data = fetch_url(self.url)

            with open(self.file_path, "wb") as fh:
                fh.write(file_data)

            return
        except Exception:
            if self.is_available:
                # fail gracefully
                logger.warning(
                    'failed to fetch kernel "%s", using local file anyway (%s)',
                    self.file_name,
                )
                return
            elif (
                os.path.isfile(self.file_path) and os.path.getsize(self.file_path) == 0
            ):
                # special case, clean up
                os.remove(self.file_path)

        raise Exception('failed to fetch kernel "{0!s}"'.format(self.file_name))

    def load(self) -> None:
        spiceypy.furnsh(self.file_path)


class SpiceKernelManager(object):
    """SPICE Kernel Manager class."""

    all_grps: dict
    all_spcs: dict

    data_path: str

    kernel_list: List[SpiceKernel]
    group_list: List[str]

    def __init__(self, json_file: str = None) -> None:
        logger = lg.getLogger(__name__)

        # clear all kernels
        spiceypy.kclear()

        self.kernel_list = []
        self.group_list = []

        base_path = os.path.join(os.path.dirname(heliosat.__file__), "spacecraft")

        self.data_path = os.getenv(
            "HELIOSAT_DATAPATH", os.path.join(os.path.expanduser("~"), ".heliosat")
        )

        logger.debug('using data path "%s"', self.data_path)

        # create kernel & data folders if necessary
        if not os.path.exists(os.path.join(self.data_path, "kernels")):
            os.makedirs(os.path.join(self.data_path, "kernels"))

        if not os.path.exists(os.path.join(self.data_path, "data")):
            os.makedirs(os.path.join(self.data_path, "data"))

        if json_file is None:
            json_file = os.path.join(base_path, "manager.json")

        json_mang = load_json(json_file)

        if json_mang["default_kernel_path"]:
            self.all_grps = load_json(
                os.path.join(base_path, json_mang["default_kernel_path"])
            )["kernels"]
        else:
            self.all_grps = {}

        if json_mang["default_spacecraft_path"]:
            self.all_spcs = load_json(
                os.path.join(base_path, json_mang["default_spacecraft_path"])
            )["spacecraft"]
        else:
            self.all_spcs = {}

        # update all groups and spacecraft
        for spacecraft in json_mang["spacecraft"]:
            if os.path.isfile(os.path.join(base_path, "{}.json".format(spacecraft))):
                self.all_grps.update(
                    load_json(
                        os.path.join(base_path, "{}.json".format(spacecraft))
                    ).get("kernels", {})
                )
                self.all_spcs.update(
                    load_json(
                        os.path.join(base_path, "{}.json".format(spacecraft))
                    ).get("spacecraft", {})
                )

        self.load_spacecraft()

    def load_group(self, kernel_group: str, force_download: bool = False) -> None:
        logger = lg.getLogger(__name__)

        logger.debug('loading kernel group "%s"', kernel_group)

        # load groups in parallel (quicker downloads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.load_kernels, urls, kernel_group, force_download)
                for urls in self.all_grps[kernel_group]
            ]

            for future in concurrent.futures.as_completed(futures):
                kernels = future.result()

                for kernel in kernels:
                    if kernel.file_name not in [_.file_name for _ in self.kernel_list]:
                        self.kernel_list.append(kernel)
                        self.kernel_list[-1].load()

        if kernel_group not in self.group_list:
            self.group_list.append(kernel_group)

    def load_groups(
        self, kernel_groups: List[str], force_download: bool = False
    ) -> None:
        for kernel_group in kernel_groups:
            self.load_group(kernel_group, force_download)

    def load_kernels(
        self, url: str, group: str, force_download: bool = False
    ) -> Union[SpiceKernel, List[SpiceKernel]]:
        # check for regex
        if url.startswith("$"):
            # figure out if kernels exist locally that match the regex, if any assume its all
            kernels = []
            local_files, _ = url_regex_files(
                url, os.path.join(self.data_path, "kernels")
            )

            if len(local_files) > 0 and not force_download:
                for local_file in local_files:
                    resolved_url = os.path.join(
                        url_dirname(url), url_basename(local_file)
                    )

                    kernel = SpiceKernel(resolved_url, self.data_path)
                    kernels.append(kernel)

                return kernels

            resolved_urls = url_regex_resolve(url)[0]

            for resolved_url in resolved_urls:
                kernel = SpiceKernel(resolved_url, self.data_path)
                kernel.prepare(force_download=force_download)
                kernels.append(kernel)

            return kernels
        else:
            kernel = SpiceKernel(url, self.data_path)
            kernel.prepare(force_download=force_download)

            return [kernel]

    def load_spacecraft(self) -> None:
        for spc_k, spc_v in self.all_spcs.items():
            aux_funcs: dict = {}

            setattr(
                heliosat,
                spc_v["class_name"],
                type(
                    spc_v["class_name"],
                    (Spacecraft,),
                    {
                        "name": spc_v["class_name"],
                        "name_naif": spc_v["body_name"],
                        "kernel_group": spc_v["kernel_group"],
                        "_json": spc_v,
                        **aux_funcs,
                    },
                ),
            )

            # allow classes to be accessed by upper case name
            if not spc_v["class_name"].isupper():
                setattr(
                    heliosat,
                    spc_v["class_name"].upper(),
                    getattr(heliosat, spc_v["class_name"]),
                )

            # runpy
            custompy = os.path.join(
                os.path.dirname(heliosat.__file__),
                "spacecraft",
                "zpy_{}.py".format(spc_k),
            )

            if os.path.isfile(custompy):
                run_path(custompy)

    def reload(self) -> None:
        # clear all kernels
        spiceypy.kclear()

        for kernel in self.kernel_list:
            kernel.load()
