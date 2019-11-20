# -*- coding: utf-8 -*-

"""spacecraft.py

Implements the Spacecraft base class and all spacecraft classes.
"""

import cdflib
import datetime
import gzip
import heliosat
import json
import logging
import multiprocessing
import numpy as np
import os
import shutil
import spiceypy

from netCDF4 import Dataset

from heliosat.caching import generate_cache_key, get_cache_entry, cache_entry_exists, \
    set_cache_entry
from heliosat.download import download_files
from heliosat.smoothing import smooth_data
from heliosat.spice import SpiceObject, spice_load, transform_frame
from heliosat.util import string_to_datetime, urls_build, urls_resolve


class Spacecraft(SpiceObject):
    """Base Spacecraft class.
    """
    spacecraft = None

    def __init__(self, name, body_name, **kwargs):
        """Summary

        Parameters
        ----------
        name : str
            Spacecraft name as defined in the spacecraft.json file.
        body_name : str
            SPICE body name assosicated with the given spacecraft.

        Other Parameters
        ----------------
        kernel_group: str
            Spacecraft kernel group as defined in the spacecraft.json file, by default None.

        Raises
        ------
        NotImplementedError
            If spacecraft is not implemented (not defined in the spacecraft.json file).
        """
        logger = logging.getLogger(__name__)

        super(Spacecraft, self).__init__(name, body_name, **kwargs)

        if heliosat._spice.get("spacecraft", None) is None:
            logger.info("loading available spacecraft")

            module_path = os.path.dirname(heliosat.__file__)

            with open(os.path.join(module_path, "json/spacecraft.json")) as json_file:
                heliosat._spice["spacecraft"] = json.load(json_file)

            self.spacecraft = heliosat._spice["spacecraft"][name]
        elif name in heliosat._spice["spacecraft"]:
            self.spacecraft = heliosat._spice["spacecraft"][name]
        else:
            logger.exception("spacecraft \"%s\" is not implemented", name)
            raise NotImplementedError("spacecraft \"%s\" is not implemented", name)

        if kwargs.get("kernel_group", None) is None and self.spacecraft.get("kernel_group", None):
            spice_load(self.spacecraft["kernel_group"])

    def get_data(self, t, data_key, **kwargs):
        """Get processed data for type "data_key" at specified datetimes t.

        Parameters
        ----------
        t : list[datetime.datetime]
            Evaluation datetimes.
        data_key : str
            Data key.

        Other Parameters
        ----------------
        cache: bool
            Cache data, by default False.
        cache_raw: bool
            Store raw data files, by default False.
        extra_columns: str
            Read extra columns from data files, by default None.
        frame: str
            Data reference frame, by default None.
        frame_interval: float
            For large data arrays, evaluate reference frame every "frame_interval" seconds,
            by defautl None.
        remove_nans: bool
            Remove NaN values from data array, by default False.
        return_datetimes: bool
            Return datetimes instead of timestamps , by default False.
        smoothing: str
            Smoothing method, by default None.
        smoothing_scale: float
            Smoothing scale in seconds, by default 300.

        Returns
        -------
        (list[float], np.ndarray)
            Evaluation datetimes as timestamps & processed data array.

        Raises
        ------
        KeyError
            If frame is specified but data has no associated frame.
        """
        logger = logging.getLogger(__name__)

        data_key = self.resolve_data_key(data_key)

        identifiers = {
            "data_key": data_key,
            "spacecraft": self.name,
            "times": [_t.timestamp() for _t in t]
        }

        identifiers.update(kwargs)

        cache = kwargs.pop("cache", False)
        frame = kwargs.pop("frame", None)
        frame_interval = kwargs.pop("frame_interval", None)
        remove_nans = kwargs.pop("remove_nans", False)
        return_datetimes = kwargs.pop("return_datetimes", False)
        smoothing = kwargs.get("smoothing", None)

        smoothing_dict = {"smoothing": smoothing}
        for key in dict(kwargs):
            if "smoothing" in key:
                smoothing_dict[key] = kwargs.pop(key)

        # fetch cached entry if selected and available
        if cache:
            cache_key = generate_cache_key(identifiers)

            if cache_entry_exists(cache_key):
                return get_cache_entry(cache_key)

        time, data = self.get_data_raw(t[0], t[-1], data_key, **kwargs)

        if smoothing:
            time, data = smooth_data(t, time, data, **smoothing_dict)

        # remove NaN's which may persist after smoothing
        if remove_nans:
            nan_mask = np.invert(np.isnan(data[:, 0]))
            time = time[nan_mask]
            data = data[nan_mask]

        if frame and "frame" in self.spacecraft["data"][data_key]:
            data = transform_frame(time, data, self.spacecraft["data"][data_key]["frame"], frame,
                                   frame_interval=frame_interval)
        elif frame and not self.spacecraft["data"][data_key].get("frame"):
            logger.exception("no reference frame defined for data_key \"%s\"", data_key)
            raise KeyError("no reference frame defined for data_key \"%s\"", data_key)

        if cache and not cache_entry_exists(cache_key):
            logger.info("generating cache entry \"%s\"", cache_key)
            set_cache_entry(cache_key, (time, data))

        if return_datetimes:
            time = [datetime.datetime.fromtimestamp(_ts) for _ts in time]

        return time, data

    def get_data_raw(self, range_start, range_end, data_key, **kwargs):
        """Get raw data for type "data_key" within specified time range.

        Parameters
        ----------
        range_start : datetime.datetime
            Time range start datetime.
        range_end : datetime.datetime
            Time range end datetime.
        data_key : str
            Data key.

        Other Parameters
        ----------------
        cache_raw: bool
            Store raw data files, by default False.
        extra_columns: str
            Read extra columns from data files, by default None.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Evaluation datetimes as timestamps & raw data array.

        Raises
        ------
        KeyError
            If data_key does not exist.
        """
        logger = logging.getLogger(__name__)

        data_key = self.resolve_data_key(data_key)

        raw_files = self.get_data_files(range_start, range_end, data_key)

        logger.info("using %s files to generate "
                    "data in between %s - %s", len(raw_files), range_start, range_end)

        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count() * 2, len(raw_files)))

        kwargs_read = dict(self.spacecraft["data"][data_key])

        if "extra_columns" in kwargs:
            kwargs_read["columns"].extend(kwargs["extra_columns"])

        results = pool.starmap(read_file, [(raw_files[i], range_start, range_end,
                                            kwargs_read)
                                           for i in range(len(raw_files))])

        i = 0
        while True:
            if results[i][0] is None:
                results.pop(i)
                i = i - 1

            i += 1

            if i >= len(results):
                break

        # concatenate data
        lengths = [len(result[0]) for result in results]
        columns = len(results[0][1][0])

        time = np.empty(sum(lengths), dtype=np.float64)
        data = np.empty((sum(lengths), columns), dtype=np.float32)

        for i in range(0, len(results)):
            if len(results[i][0]) > 0:
                time[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][0]
                data[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][1]

        # delete raw data files if required (must be explicitly set)
        if not kwargs.get("cache_raw", True):
            for file in raw_files:
                os.remove(file)

        return time, data

    def get_data_files(self, range_start, range_end, data_key):
        """Get raw data file paths for type "data_key" within specified time range.

        Parameters
        ----------
        range_start : datetime.datetime
            Time range start datetime.
        range_end : datetime.datetime
            Time range end datetime.
        data_key : str
            Data key.

        Returns
        -------
        list
            Raw data file paths.
        """
        logger = logging.getLogger(__name__)

        if range_start < self.mission_start or range_end > self.mission_end:
            logger.exception("invalid time range (must be within %s - %s)",
                             string_to_datetime(self.mission_start),
                             string_to_datetime(self.mission_end))
            raise ValueError("invalid time range (must be within %s - %s)",
                             string_to_datetime(self.mission_start),
                             string_to_datetime(self.mission_end))

        data_path = os.path.join(heliosat._paths["data"], data_key)

        urls = urls_build(self.spacecraft["data"][data_key]["urls"], range_start, range_end,
                          versions=self.spacecraft["data"][data_key].get("versions", None))

        # resolve regex urls if required
        if self.spacecraft["data"][data_key].get("use_regex", False):
            urls = urls_resolve(urls)

        # some files are archives, skip the download if the extracted versions exist
        files_extracted = []

        for url in list(urls):
            if url.endswith(".gz"):
                file = ".".join(url.split("/")[-1].split(".")[:-1])

                if os.path.exists(os.path.join(data_path, file)):
                    logger.info("skipping download for \"%s\" as extracted version already exists",
                                url)

                    urls.remove(url)
                    files_extracted.append(os.path.join(data_path, file))

        download_files(urls, data_path, logger=logger)

        files = [
            os.path.join(data_path, urls[i].split("/")[-1])
            for i in range(0, len(urls))
        ]

        # extract new archives and add old ones to the list
        files.extend(files_extracted)

        for file in list(files):
            if file.endswith(".gz"):
                with gzip.open(file, "rb") as file_gz:
                    with open(".".join(file.split(".")[:-1]), "wb") as file_extracted:
                        shutil.copyfileobj(file_gz, file_extracted)

                os.remove(file)
                files.remove(file)
                files.append(".".join(file.split(".")[:-1]))
            if not os.path.isfile(file):
                files.remove(file)

        return files

    @property
    def data_keys(self):
        return list(self.spacecraft["data"].keys())

    @property
    def mission_start(self):
        return string_to_datetime(self.spacecraft["mission_start"])

    @property
    def mission_end(self):
        if "mission_end" in self.spacecraft:
            return string_to_datetime(self.spacecraft.get("mission_end"))
        else:
            return datetime.datetime.now()

    def resolve_data_key(self, data_key):
        """Replace data_key with actual key in case it is an alternative name.

        Parameters
        ----------
        data_key : str
            Data key (alternative).

        Returns
        -------
        str
            Actual data key.

        Raises
        ------
        ValueError
            If invalid time range is given.
        """
        # check if data_key exists, or check for general name
        if data_key not in self.spacecraft["data"]:
            key_resolved = False

            for key in self.spacecraft["data"]:
                alt_names = self.spacecraft["data"][key].get("alt_names", [])

                if data_key in alt_names:
                    data_key = key
                    key_resolved = True
                    break

            if not key_resolved:
                raise KeyError("data_key \"%s\" is not defined", data_key)

        return data_key


class DSCOVR(Spacecraft):
    def __init__(self):
        super(DSCOVR, self).__init__("dscovr", body_name="EARTH")


class MES(Spacecraft):
    def __init__(self):
        super(MES, self).__init__("messenger", body_name="MESSENGER")


class PSP(Spacecraft):
    def __init__(self):
        from spiceypy.utils.support_types import SpiceyError

        super(PSP, self).__init__("psp", body_name="SPP")

        try:
            spiceypy.bodn2c("SPP_SPACECRAFT")
        except SpiceyError:
            # kernel fix
            spiceypy.boddef("SPP_SPACECRAFT", spiceypy.bodn2c("SPP"))


class STA(Spacecraft):
    def __init__(self):
        super(STA, self).__init__("stereo_ahead", body_name="STEREO AHEAD")


class STB(Spacecraft):
    def __init__(self):
        super(STB, self).__init__("stereo_behind", body_name="STEREO BEHIND")


class VEX(Spacecraft):
    def __init__(self):
        super(VEX, self).__init__("venus_express", body_name="VENUS EXPRESS")


class WIND(Spacecraft):
    def __init__(self):
        super(WIND, self).__init__("wind", body_name="EARTH")


def read_file(file_path, range_start, range_end, kwargs):
    """Worker function for reading data files.

    Parameters
    ----------
    file_path : str
        File path.
    range_start : datetime.datetime
        Time range start datetime.
    range_end : datetime.datetime
        Time range end datetime.
    kwargs : dict
        Additional parameters (dict as defined in spacecraft.json)

    Returns
    -------
    (np.ndarray, np.ndarray)
        Datetimes as timestamps and raw data array

    Raises
    ------
    NotImplementedError
        If data format is not supported.
    NotImplementedError
        If time format is not supported (pds3 format only).
    """
    logger = logging.getLogger(__name__)

    columns = kwargs.get("columns")
    format = kwargs.get("format")
    values = kwargs.get("values", None)

    if format == "cdf":
        file = cdflib.CDF(file_path)

        # offset between python datetimes and cdf epochs
        time = cdflib.epochs.CDFepoch.unixtime(file.varget(columns[0]), to_np=True)
        data = [np.array(file.varget(key.split(":")[0]), dtype=np.float32) for key in columns[1:]]
    elif format == "netcdf":
        file = Dataset(file_path, "r")

        time = np.array([t / 1000 for t in file.variables[columns[0]][...]])
        data = [np.array(file[key][:], dtype=np.float32) for key in columns[1:]]
    elif format == "pds3":
        skip_rows = kwargs.get("skip_rows", 0)
        time_format_type = kwargs.get("time_format_type", "string")
        time_format = kwargs.get("time_format", "%Y-%m-%dT%H:%M:%S.%f")
        time_offset = kwargs.get("time_offset", 0)

        def decode_string(string, format):
            # fix datetimes with "60" as seconds
            if string.endswith("60.000"):
                # TODO: fix :17 (only works for one specific format)
                string = "{0}59.000".format(string[:17])

                return datetime.datetime.strptime(string, format).timestamp() + 1
            else:
                return datetime.datetime.strptime(string, format).timestamp()

        if time_format_type == "string":
            file = np.loadtxt(file_path, skiprows=skip_rows, encoding="latin1",
                              converters={0: lambda s: decode_string(s, time_format)})

            time = file[:, columns[0]]
        elif time_format_type == "timestamp":
            file = np.loadtxt(file_path, skiprows=skip_rows)

            time = file[:, columns[0]] + time_offset
        else:
            logger.exception("time_format_type \"%s\" is not implemented", time_format_type)
            raise NotImplementedError("time_format_type \"%s\" is not implemented",
                                      time_format_type)

        data = [np.array(file[:, key], dtype=np.float32) for key in columns[1:]]
    else:
        logger.exception("format \"%s\" is not implemented", format)
        raise NotImplementedError("format \"%s\" is not implemented", format)

    # fix some odd shapes
    for i in range(len(data)):
        if data[i].ndim == 1:
            data[i] = data[i].reshape(-1, 1)

    # discard unneeded dimensions
    if isinstance(columns[0], str):
        for i in range(0, len(columns[1:])):
            if ":" in columns[i + 1]:
                data[i] = data[i][:, 0:int(columns[i + 1].split(":")[1])]

    mask = np.where((time > range_start.timestamp()) & (time < range_end.timestamp()))[0]

    if len(mask) > 1:
        time_part = np.squeeze(time[mask])

        for i in range(len(data)):
            if i == 0:
                data_part = data[0][mask]
            else:
                data_part = np.hstack((data_part, data[i][mask]))

        data_part = np.squeeze(data_part)

        # remove values outside valid range
        if values:
            valid_indices = np.where(
                np.array(data_part[:, 0] > values[0]) & np.array(data_part[:, 0] < values[1])
            )

            time_part = time_part[valid_indices]
            data_part = data_part[valid_indices]

        # sort time array
        sort_mask = np.argsort(time_part)
        time_part = time_part[sort_mask]
        data_part = data_part[sort_mask]
    else:
        time_part = None
        data_part = None

    return time_part, data_part
