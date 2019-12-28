# -*- coding: utf-8 -*-

"""helcats.py

Basic integration for HELCATS. The current focus is on ICMECAT & LINKCAT (coronal mass ejections).
"""

import datetime
import heliosat
import json
import logging
import os

from heliosat.download import download_files
from heliosat.util import string_to_datetime


class Event:
    _observer = None

    def get_data(self, data_key="mag", buffer=12, resolution=5, **kwargs):
        """Wrapper for Spacecraft.get_data.

        Parameters
        ----------
        data_key : str
            Data key.
        buffer : float
            Buffer in hours to append on both sides of the time interval.
        resolution : float
            Time resolution in minutes.

        Other Parameters
        ----------------
        cache: bool
            Cache data, by default False.
        cache_raw: bool
            Store raw data files, by default False.
        columns: list[str]
            Read columns instead of default, by default None.
        extra_columns: list[str]
            Read extra columns ontop of default, by default None.
        frame: str
            Data reference frame, by default None.
        frame_cadence: float
            Evaluate frame transformation matrix every "frame_cadence" seconds instead of at very
            time point (significant speed up).
        return_datetimes: bool
            Return datetimes instead of timestamps, by default False.
        smoothing: str
            Smoothing method, by default None.
        smoothing_scale: float
            Smoothing scale in seconds, by default 300.

        Returns
        -------
        (list[float], np.ndarray)
            Evaluation datetimes as timestamps & processed data array.
        """
        t0 = self.time_start - datetime.timedelta(hours=buffer)
        t1 = self.time_end + datetime.timedelta(hours=buffer)

        steps = int((t1.timestamp() - t0.timestamp()) / 60 / resolution)

        ts = [t0 + datetime.timedelta(minutes=i * resolution) for i in range(steps)]

        cache = kwargs.pop("cache", True)
        rdt = kwargs.pop("return_datetimes", True)
        sm = kwargs.pop("smoothing", "kernel")
        sms = kwargs.pop("smoothing_scale", 300)

        return self.observer.get_data(ts, data_key, cache=cache, return_datetimes=rdt, smoothing=sm,
                                      smoothing_scale=sms, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def observer(self):
        if not self._observer:
            self._observer = self._observer_cls()

        return self._observer


class HELCATS_EVENT(Event):
    def __init__(self, skip_download=False):
        """Initialize HELCATS event.

        Parameters
        ----------
        skip_download: bool, optional
            Skip catalog downloads (requires catalogs to exist locally), by default False.
        """
        logger = logging.getLogger(__name__)

        if heliosat._helcats is None:
            logger.info("running helcats_init")
            helcats_init(skip_download)


class ICMECAT_EVENT(HELCATS_EVENT):
    def __init__(self, event, skip_download=False):
        """Initialize ICMECAT event.

        Parameters
        ----------
        event: str
            Event name.
        skip_download: bool, optional
            Skip catalog downloads (requires catalogs to exist locally), by default False.
        """
        super(ICMECAT_EVENT, self).__init__(skip_download)

        # find event
        event_found = False

        for entry in heliosat._helcats["catalogs"]["icmecat"]["data"]:
            if event == entry[0]:
                event_found = True

                data = entry

        if not event_found:
            raise KeyError("event %s could not be found in ICMECAT", event)

        self._name = entry[4]

        if data[1].lower() in ["dscovr"]:
            self._observer_cls = heliosat.DSCOVR
        elif data[1].lower() in ["mes", "messenger"]:
            self._observer_cls = heliosat.MES
        elif data[1].lower() in ["psp"]:
            self._observer_cls = heliosat.PSP
        elif data[1].lower() in ["stereo_a", "stereo-a"]:
            self._observer_cls = heliosat.STA
        elif data[1].lower() in ["stereo_b", "stereo-b"]:
            self._observer_cls = heliosat.STB
        elif data[1].lower() in ["vex"]:
            self._observer_cls = heliosat.VEX
        elif data[1].lower() == "wind":
            self._observer_cls = heliosat.WIND
        else:
            raise NotImplementedError("satellite %s is not implemented", data[1])

        try:
            self.time_end = string_to_datetime(data[5])
        except ValueError:
            self.time_end = string_to_datetime(data[4]) + datetime.timedelta(hours=6)

        self.time_start = string_to_datetime(data[2])


def helcats_init(skip_download=False):
    """Initialize HELCATS catalogs.

    Parameters
    ----------
    skip_download : bool, optional
        Skip catalog downloads (requires catalogs to exist locally), by default False.
    """
    logger = logging.getLogger(__name__)

    helcats_path = heliosat._paths["helcats"]

    helcats_catalogs_path = os.path.join(os.path.dirname(heliosat.__file__), "json/helcats.json")

    heliosat._helcats = {"catalogs": {}}

    # json/helcats.json
    logger.info("loading available catalogs")

    with open(helcats_catalogs_path) as json_file:
        json_catalogs = json.load(json_file)

    if not os.path.exists(helcats_path):
        os.makedirs(helcats_path)

    # load catalogs
    if not skip_download:
        download_files([json_catalogs[key] for key in json_catalogs], helcats_path, logger=logger)

    for catalog in json_catalogs:
        catalog_url = json_catalogs[catalog]
        catalog_path = os.path.join(helcats_path, catalog_url.split("/")[-1])

        logger.info("loading %s catalog", catalog)

        with open(catalog_path) as json_file:
            json_catalog = json.load(json_file)

        heliosat._helcats["catalogs"][catalog] = json_catalog
