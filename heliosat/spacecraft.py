# -*- coding: utf-8 -*-

"""spacecraft.py

Implements the Spacecraft base class. All implemented spacecraft classes inherit from the base
Spacecraft class which defines basic data handling routines.
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

from concurrent.futures import ProcessPoolExecutor
from heliosat.caching import generate_cache_key, get_cache_entry, cache_entry_exists, \
    set_cache_entry
from heliosat.download import download_files
from heliosat.smoothing import smooth_data
from heliosat.spice import SpiceObject, spice_load, transform_frame
from heliosat.util import datetime_to_string, string_to_datetime, urls_resolve
from netCDF4 import Dataset
from itertools import compress


class Spacecraft(SpiceObject):
    """Base Spacecraft class.
    """
    spacecraft = None

    def __init__(self, name, body_name, **kwargs):
        """Initialize Spacecraft object. This just fetches the spacecraft.json entry and loads any
        spacecraft specific SPICE kernels.

        Parameters
        ----------
        name : str
            Spacecraft name as defined in the spacecraft.json file.
        body_name : str
            SPICE body name assosicated with the given spacecraft. In some cases a different body
            is given with a similar orbital trajectory (e.g. Earth for DSCOVR/Wind).

        Other Parameters
        ----------------
        kernel_group: str
            Spacecraft kernel group as defined in the spacecraft.json file, by default None.
        skip_download: bool
            Skip kernel downloads (requires kernels to exist locally), by default False.

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
            spice_load(self.spacecraft["kernel_group"],
                       skip_download=kwargs.get("skip_download", False))

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
        logger = logging.getLogger(__name__)

        data_key = self.resolve_data_key(data_key)

        identifiers = {
            "data_key": data_key,
            "spacecraft": self.name,
            "times": [_t.timestamp() for _t in t]
        }

        identifiers.update(kwargs)

        # clean up kwargs so it can be passed on to get_data_raw
        cache = kwargs.get("cache", False)
        return_datetimes = kwargs.pop("return_datetimes", False)
        smoothing = kwargs.get("smoothing", None)

        if cache:
            # fetch cached entry
            cache_key = generate_cache_key(identifiers)

            if cache_entry_exists(cache_key):
                return get_cache_entry(cache_key)

        time, data = self.get_data_raw(t[0], t[-1], data_key, **kwargs)

        if smoothing:
            # smooth data
            # all parameters that include "smoothing" are passed onto the smoothing function
            smoothing_dict = {"smoothing": smoothing}

            logger.info("smoothing data (%s)", smoothing)

            for key in dict(kwargs):
                if "smoothing" in key:
                    smoothing_dict[key] = kwargs.pop(key)

            time, data = smooth_data(t, time, data, **smoothing_dict)
        else:
            # no smoothing, choose closest time values
            _time = []
            _data = []

            for _t in t:
                pass

        if return_datetimes:
            _time = list(time)

            for i in range(len(_time)):
                if _time[i] != np.nan:
                    _time[i] = datetime.datetime.fromtimestamp(time[i])

            time = _time

        if cache and not cache_entry_exists(cache_key):
            # save cache entry
            logger.info("generating cache entry \"%s\"", cache_key)
            set_cache_entry(cache_key, (time, data))

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
            Store raw data files, by default True.
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

        frame = kwargs.get("frame", None)
        frame_cadence = kwargs.get("frame_cadence", None)

        files, file_versions = self.get_data_files(range_start, range_end, data_key)

        logger.info("using %s files to generate "
                    "data in between %s - %s", len(files), range_start, range_end)

        max_workers = min(multiprocessing.cpu_count() * 2, len(files))

        # setup columns, ~ is a placeholder for the default columns
        columns = kwargs.get("columns", ["~"])
        columns.extend(kwargs.get("extra_columns", []))

        kernels = heliosat._spice["kernels_loaded"]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(files[i], file_versions[i], range_start, range_end, columns, frame,
                     frame_cadence, kernels)
                    for i in range(len(files))]

            futures = executor.map(read_task, args)

        results = [_ for _ in futures if _]

        time_all = np.concatenate([_[0] for _ in results])
        data_all = np.concatenate([_[1] for _ in results])

        # delete raw data files if required (must be explicitly set)
        if not kwargs.get("cache_raw", True):
            for file in files:
                os.remove(file)

        if kwargs.get("return_datetimes", False):
            _time = list(time_all)

            for i in range(len(_time)):
                if _time[i] != np.nan:
                    _time[i] = datetime.datetime.fromtimestamp(_time[i])

            time_all = _time

        return time_all, data_all

    def get_data_files(self, range_start, range_end, data_key, return_versions=False):
        """Get raw data file paths for type "data_key" within specified time range.

        Parameters
        ----------
        range_start : datetime.datetime
            Time range start datetime.
        range_end : datetime.datetime
            Time range end datetime.
        data_key : str
            Data key.
        return_versions : bool
            Return file version information (as list index).

        Returns
        -------
        list
            Raw data file paths.
        """
        logger = logging.getLogger(__name__)

        if range_end < range_start:
            raise ValueError("starting date must be after final date")

        data_key = self.resolve_data_key(data_key)
        data_path = os.path.join(heliosat._paths["data"], data_key)

        urls = []
        versions = []

        # adjust ranges slightly
        range_start -= datetime.timedelta(hours=range_start.hour, minutes=range_start.minute,
                                          seconds=range_start.second)
        if range_end.hour == 0 and range_end.minute == 0 and range_end.second == 0:
            range_end -= datetime.timedelta(seconds=1)

        # build url for each day in range (this assumes all files are daily)
        for day in [range_start + datetime.timedelta(days=i)
                    for i in range((range_end - range_start).days + 1)]:
            url = self.spacecraft["data_keys"][data_key]["urls"]

            url = url.replace("{YYYY}", str(day.year))
            url = url.replace("{YY}", "{0:02d}".format(day.year % 100))
            url = url.replace("{MM}", "{:02d}".format(day.month))
            url = url.replace("{MONTH}", day.strftime("%B")[:3].upper())
            url = url.replace("{DD}", "{:02d}".format(day.day))
            url = url.replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

            doym1 = datetime.datetime(day.year, day.month, 1)

            if day.month == 12:
                doym2 = datetime.datetime(day.year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                doym2 = datetime.datetime(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

            url = url.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
            url = url.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

            # insert version if required
            version = self.get_data_version(data_key, day)

            for i in range(len(version.get("identifiers", []))):
                url = url.replace("{{V{0}}}".format(i), version["identifiers"][i])

            urls.append(url)
            versions.append(version)

        # resolve regex urls if required
        if self.spacecraft["data_keys"][data_key].get("use_regex", False):
            urls = urls_resolve(urls)

        # some files are archives, skip the download if the extracted versions exist
        files_extracted = []
        versions_extracted = []

        for url in list(urls):
            if url.endswith(".gz"):
                file = ".".join(url.split("/")[-1].split(".")[:-1])

                if os.path.exists(os.path.join(data_path, file)):
                    logger.info("skipping download for \"%s\" as extracted version already exists",
                                url)

                    # remove url & version, and append extracted file at end
                    rindex = urls.index(url)

                    files_extracted.append(os.path.join(data_path, file))
                    versions_extracted.append(versions[rindex])

                    del urls[rindex]
                    del versions[rindex]

        if len(urls) > 0:
            results = download_files(urls, data_path, logger=logger)
            check_downloads = True
        else:
            check_downloads = False

        files = [
            os.path.join(data_path, urls[i].split("/")[-1])
            for i in range(0, len(urls))
        ]

        # extract new archives and add old ones to the list
        files.extend(files_extracted)
        versions.extend(versions_extracted)

        for file in list(files):
            if file.endswith(".gz"):
                with gzip.open(file, "rb") as file_gz:
                    with open(".".join(file.split(".")[:-1]), "wb") as file_extracted:
                        shutil.copyfileobj(file_gz, file_extracted)

                os.remove(file)
                rindex = files.index(file)
                files[rindex] = ".".join(file.split(".")[:-1])

        if check_downloads:
            return list(compress(files, results)), list(compress(versions, results))
        else:
            return files, versions

    def get_data_version(self, data_key, time=None):
        """Get version information from type "data_key" at specific time.

        Parameters
        ----------
        data_key : str
            Data key.
        time : datetime.datetime
            Version time, by default None.

        Returns
        -------
        dict
            Version information.

        Raises
        ------
        RuntimeError
            If no version is defined at specified time.
        """
        data_key = self.resolve_data_key(data_key)

        versions = self.spacecraft["data_keys"][data_key]["versions"]

        ver_found = False

        # get version defaults
        selected_version = dict(self.spacecraft["data_keys"][data_key]["version_default"])

        if time is None:
            return selected_version

        for version in versions:
            if string_to_datetime(version["version_start"]) <= time < \
               string_to_datetime(version["version_end"]):
                selected_version.update(version)

                return selected_version

        if not ver_found:
            raise RuntimeError("no version found for data key \"%s\" at time %s", data_key,
                               datetime_to_string(time))

    def get_data_columns(self, data_key, time=None, show_all=False):
        """Get data columns from type "data_key" at specific time.

        Parameters
        ----------
        data_key : str
            Data key.
        time : datetime.datetime
            Version time, by default None.
        show_all : bool
            show non-default columns, by default False

        Returns
        -------
        dict
            Version information.

        Raises
        ------
        RuntimeError
            If no version is defined at specified time.
        """
        version = self.get_data_version(data_key, time)

        return [c["name"] for c in version["columns"] if (c["default"] | show_all)]

    @property
    def data_keys(self):
        return list(self.spacecraft["data_keys"].keys())

    def resolve_data_key(self, data_key):
        """Replace data_key with actual key in case it is an alternative name.

        Parameters
        ----------
        data_key : str
            Data key (alternative).

        Returns
        -------
        str
            Data key.

        Raises
        ------
        ValueError
            If invalid time range is given.
        """
        # check if data_key exists, or check for general name
        if data_key not in self.spacecraft["data_keys"]:
            key_resolved = False

            for key in self.spacecraft["data_keys"]:
                alt_names = self.spacecraft["data_keys"][key].get("alt_keys", [])

                if data_key in alt_names:
                    data_key = key
                    key_resolved = True
                    break

            if not key_resolved:
                raise KeyError("data_key \"%s\" is not defined", data_key)

        return data_key


def read_task(args):
    """Wrapper for reading data files.

    Parameters
    ----------
    args : tuple
        Function arguments as tuple.

    Returns
    -------
    (list[float], np.ndarray)
        Evaluation datetimes as timestamps & processed data array.

    Raises
    ------
    KeyError
        If a data column is invalid.
    NotImplementedError
        If time format for text files is not supported.
    """
    (file_path, file_version, range_start, range_end, columns, frame, frame_cadence, kernels) = args

    column_dicts = []

    # resolve default columns
    if columns[0] == "~":
        default_columns = []

        for column in file_version["columns"]:
            if column.get("default", False):
                default_columns.append(column["name"])

        if len(columns) > 1:
            default_columns.extend(columns[1:])

        columns = default_columns

    # resolve alternative columns and populate column_dicts
    for i in range(len(columns)):
        valid_column = False

        for j in range(len(file_version["columns"])):
            column = file_version["columns"][j]

            if columns[i] != column["name"] and columns[i] in column["alt_names"]:
                columns[i] = column["name"]
                column_dicts.append(column)
                valid_column = True

                break
            elif columns[i] == column["name"]:
                column_dicts.append(column)
                valid_column = True

                break

        if not valid_column:
            raise KeyError("data column \"%s\" is invalid", columns[i])

    if "_cdf" in file_version["format"]:
        time, data = read_cdf_task(file_path, range_start, range_end, file_version, column_dicts,
                                   cdf_type=file_version["format"])
    elif file_version["format"] == "text":
        time, data = read_text_task(file_path, range_start, range_end, file_version, column_dicts)
    else:
        raise NotImplementedError("format \"%s\" is not implemented", file_version["format"])

    # time mask
    time_mask = np.where((time > range_start.timestamp()) & (time < range_end.timestamp()))[0]

    if len(time_mask) == 1:
        raise NotImplementedError
    if len(time_mask) > 1:
        time = time[time_mask]

        # process data
        for i in range(len(data)):
            column = column_dicts[i]

            data_entry = data[i][time_mask]

            # filter values outside of range
            valid_range = column.get("valid_range", None)

            if valid_range:
                data_entry = np.where((data_entry > valid_range[0]) & (data_entry < valid_range[1]),
                                      data_entry, np.nan)

            sort_mask = np.argsort(time)
            time = time[sort_mask]
            data_entry = data_entry[sort_mask]

            if data_entry.ndim == 1:
                data_entry = data_entry.reshape((-1, 1))

            if frame and "frame" in column and frame != column.get("frame", None):
                heliosat.spice.spice_init()
                heliosat.spice.spice_reload(kernels)

                data_entry = transform_frame(time, data_entry, column["frame"], frame,
                                             frame_cadence)

            data[i] = data_entry

        return time, np.concatenate(data, axis=1)
    else:
        return None


def read_cdf_task(file_path, range_start, range_end, version_dict, column_dicts, cdf_type):
    """Worker function for reading cdf data files.

    Parameters
    ----------
    file_path : str
        File path.
    range_start : datetime.datetime
        Time range start datetime.
    range_end : datetime.datetime
        Time range end datetime.
    version_dict : dict
        Version information.
    column_dicts : list[dict]
        Data column information.
    cdf_type : str
        CDF file type (cdf, netcdf4).

    Returns
    -------
    (list[float], np.ndarray)
        Evaluation datetimes as timestamps & processed data array.
    """
    time_dict = version_dict["time"]
    logger = logging.getLogger(__name__)

    try:
        if cdf_type == "nasa_cdf":
            file = cdflib.CDF(file_path)
            epochs = file.varget(time_dict["key"])

            # fix for some cdf files that have 0 entries
            if np.sum(epochs == 0) > 0:
                null_filter = (epochs != 0)
                epochs = epochs[null_filter]
            else:
                null_filter = None
            time = cdflib.epochs.CDFepoch.unixtime(epochs, to_np=True)
            data = []

            for column in column_dicts:
                key = column["key"]

                if isinstance(key, str):
                    indices = column.get("indices", None)

                    if indices is not None:
                        indices = np.array(indices)
                        data.append(np.array(file.varget(key)[:, indices]))
                    else:
                        data.append(np.array(file.varget(key)))
                elif isinstance(key, list):
                    data.append(np.stack(arrays=[np.array(file.varget(k))
                                                 for k in key], axis=1))
                else:
                    raise NotImplementedError

            if null_filter is not None:
                for i in range(len(data)):
                    data[i] = data[i][null_filter]
        elif cdf_type == "net_cdf4":
            file = Dataset(file_path, "r")

            time = np.array([t / 1000 for t in file.variables[time_dict["key"]][...]])
            data = []

            for column in column_dicts:
                key = column["key"]

                if isinstance(key, str):
                    indices = column.get("indices", None)

                    if indices is not None:
                        indices = np.array(indices)
                        data.append(np.array(file[key][:, indices]))
                    else:
                        data.append(np.array(file[key][:]))
                elif isinstance(key, list):
                    data.append(np.stack(arrays=[np.array(file[k][:])
                                                 for k in key], axis=1))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError("CDF type \"%s\" is not supported", cdf_type)

        return time, data
    except Exception as err:
        logger.error("failed to read file \"%s\" (%s)", file_path, err)
        raise Exception


def read_text_task(file_path, range_start, range_end, version_dict, column_dicts):
    """Worker function for reading text files.

    Parameters
    ----------
    file_path : str
        File path.
    range_start : datetime.datetime
        Time range start datetime.
    range_end : datetime.datetime
        Time range end datetime.
    version_dict : dict
        Version information.
    column_dicts : list[dict]
        Data column information.

    Returns
    -------
    (list[float], np.ndarray)
        Evaluation datetimes as timestamps & processed data array.
    """
    logger = logging.getLogger(__name__)

    skip_rows = version_dict.get("skip_rows", 0)

    time_format = version_dict["time"]["format"]
    time_index = version_dict["time"]["index"]

    delimiter = version_dict.get("delimiter", None)

    def decode(string, format):
        if string.endswith("60.000"):
            string = "{0}59.000".format(string[:-6])

            return datetime.datetime.strptime(string, format).timestamp() + 1
        else:
            return datetime.datetime.strptime(string, format).timestamp()

    converters = {}

    skip_cols = version_dict.get("skip_columns")

    for col in skip_cols:
        converters[col] = lambda string: -1

    if isinstance(time_format, str):
        converters[0] = lambda string: decode(string, time_format)

        file = np.loadtxt(file_path, skiprows=skip_rows, encoding="latin1", delimiter=delimiter,
                          converters=converters)

        time = file[:, time_index]
    elif isinstance(time_format, int):
        file = np.loadtxt(file_path, skiprows=skip_rows, converters=converters)

        time = file[:, time_index] + time_format
    else:
        logger.exception("time_format \"%s\" is not implemented", type(time_format))
        raise NotImplementedError("time_format \"%s\" is not implemented", time_format)

    data = []

    for column in column_dicts:
        indices = column["indices"]

        if isinstance(indices, int):
            indices = column.get("indices", None)
            indices = np.array(indices)
            data.append(np.stack([np.array(file[:, index]) for index in indices]))

        elif isinstance(indices, list):
            data.append(np.stack([np.array(file[:, index])
                                  for index in indices], axis=1))
        else:
            raise NotImplementedError

    return time, data
