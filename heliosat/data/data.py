# -*- coding: utf-8 -*-

import datetime
import gzip
import logging
import multiprocessing
import numpy as np
import os
import re
import requests
import shutil

from bs4 import BeautifulSoup
from netCDF4 import Dataset
from spacepy import pycdf

from heliosat.util import download_files


class DataObject(object):
    def get_data_files(self, start_time, stop_time, data):
        """Resolve list of data files that contain the raw data for the time range
        [start_time, stop_time].

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time
        data: str
            data type

        Returns
        -------
        list
            data file list
        """
        logger = logging.getLogger(__name__)

        urls = _build_urls(self._data[data]["remote_files"], self._data[data].get("versions", None),
                           start_time, stop_time)

        # resolve regex expressions in urls
        if self._data[data].get("use_regex", False):
            urls = _resolve_urls(urls)

        # skip extracted files
        urls_skipped = []

        for url in list(urls):
            if url.endswith(".gz"):
                file_path = ".".join(url.split("/")[-1].split(".")[:-1])

                if os.path.exists(os.path.join(self._data_folder, file_path)):
                    logger.info("found {0}, skipping download".format(file_path))
                    urls.remove(url)
                    urls_skipped.append(os.path.join(self._data_folder, file_path))

        download_files(urls, self._data_folder, logger=logger)

        files = [
            os.path.join(self._data_folder, urls[i].split("/")[-1])
            for i in range(0, len(urls))
        ]

        files.extend(urls_skipped)

        # extract archived files
        for file in files:
            if file.endswith(".gz"):
                with gzip.open(file, "rb") as file_gz:
                    with open(".".join(file.split(".")[:-1]), "wb") as file_new:
                        shutil.copyfileobj(file_gz, file_new)

                os.remove(file)

        # remove archive extensions
        for file in list(files):
            if file.endswith(".gz"):
                files.remove(file)
                files.append(os.path.join(self._data_folder,
                                          ".".join(file.split("/")[-1].split(".")[:-1])))
        return files

    def get_data_raw(self, start_time, stop_time, data, extra_data=None, skip_files=False):
        """Get raw data for the time range [start_time, stop_time].

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time
        data: str
            data type
        extra_keys : list
            list of extra cdf keys or columns, by default None
        skip_files : bool, optional
            skip missing files, by default False

        Returns
        -------
        np.ndarray, np.ndarray
            datetime timestamps, data
        """
        logger = logging.getLogger(__name__)

        # sanity checks
        assert start_time < stop_time, "start_time must be before stop_time"
        assert start_time >= self._start_time, "start_time must be after mission start time"

        if hasattr(self, "_stop_time"):
            assert stop_time < self._stop_time, "stop_time must be before mission stop time"

        files = self.get_data_files(start_time, stop_time, data)

        if skip_files:
            # remove missing files
            for file in list(files):
                if not os.path.exists(file):
                    files.remove(file)

            if len(files) == 0:
                raise FileNotFoundError("no raw data files found")
        else:
            for file in list(files):
                if not os.path.exists(file):
                    raise FileNotFoundError("raw data file \"{0}\" "
                                            "could not be found".format(file))

        logger.info("using {0} files to generate "
                    "data in between {1} - {2}".format(len(files), start_time, stop_time))

        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count() * 2, len(files)))

        if self._data[data]["file_format"] == "cdf":
            keys = self._data[data]["cdf_keys"]

            if extra_data:
                keys.extend(extra_data)

            results = pool.starmap(_read_cdf_file,
                                   [(files[i], start_time, stop_time, keys)
                                    for i in range(0, len(files))])
        elif self._data[data]["file_format"] == "netcdf":
            keys = self._data[data]["cdf_keys"]

            if extra_data:
                keys.extend(extra_data)

            results = pool.starmap(_read_netcdf_file, [(files[i], start_time, stop_time, keys)
                                   for i in range(0, len(files))])
        elif self._data[data]["file_format"] == "pds3":
            cols = self._data[data]["columns"]

            if extra_data:
                cols.extend(extra_data)

            results = pool.starmap(_read_pds3_file, [(files[i], start_time, stop_time,
                                                     cols,
                                                     self._data[data].get("skip_rows", 0),
                                                     self._data[data]["time_format"],
                                                     self._data[data]["time_format_data"])
                                                     for i in range(0, len(files))])

        # concatenate data
        lengths = [len(result[0]) for result in results]
        length_total = sum(lengths)
        length_data = len(results[0][1][0])

        result_ts = np.empty(length_total, dtype=np.float64)
        result_data = np.empty((length_total, length_data), dtype=np.float32)

        for i in range(0, len(results)):
            if len(results[i][0]) > 0:
                result_ts[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][0]
                result_data[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][1]

        # remove NaN values
        if "valid_range" in self._data[data]:
            valid_index = np.where(
                np.array(result_data[:, 0] > self._data[data]["valid_range"][0]) &
                np.array(result_data[:, 0] < self._data[data]["valid_range"][1])
                )

            result_ts = result_ts[valid_index]
            result_data = result_data[valid_index]

        return result_ts, result_data

    def get_ref_frame(self, data):
        """Get reference frame for specified data type.

        Parameters
        ----------
        data: str
            data type

        Returns
        -------
        str
            reference frame
        """
        return self._data[data]["reference_frame"]


def _build_urls(format_string, versions, start_time, stop_time):
    """Build url list from files for the time range [start_time, stop_time].

    Parameters
    ----------
    format_string : Union[list, str]
        format string or list
    start_time : datetime.datetime
        start time
    stop_time : datetime.datetime
        stop time

    Returns
    -------
    list
        urls
    """
    def ct(tstr):
        return datetime.datetime.strptime(tstr, "%Y-%m-%dT%H:%M:%S.%f")

    days = [start_time]

    while days[-1] < stop_time:
        days.append(days[-1] + datetime.timedelta(days=1))

    urls = []

    for day in days:
        urls.append(format_string)
        urls[-1] = urls[-1].replace("{YYYY}", str(day.year))
        urls[-1] = urls[-1].replace("{YY}", "{0:02d}".format(day.year % 100))
        urls[-1] = urls[-1].replace("{MM}", "{:02d}".format(day.month))
        urls[-1] = urls[-1].replace("{MONTH}", day.strftime("%B")[:3].upper())
        urls[-1] = urls[-1].replace("{DD}", "{:02d}".format(day.day))
        urls[-1] = urls[-1].replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

        doym1 = datetime.datetime(day.year, day.month, 1)
        doym2 = datetime.datetime(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

        urls[-1] = urls[-1].replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
        urls[-1] = urls[-1].replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

        if versions:
            found = True

            for entry in versions:
                if ct(entry["start_time"]) <= day < ct(entry["stop_time"]):
                    found = True

                    for i in range(len(entry["strings"])):
                        urls[-1] = urls[-1].replace("{{V{0}}}".format(i), entry["strings"][i])

                    break

            if not found:
                raise RuntimeError("version for {0} not found".format(day))

    return urls


def _read_cdf_file(file_path, start_time, stop_time, keys):
    """Worker function for reading cdf data files.

    Parameters
    ----------
    file_path : str
        path to cdf file
    start_time : datetime.datetime
        starting time
    stop_time : datetime.datetime
        end time
    keys: tuple
        cdf keys (first key is assumed to be the epoch key, all others data keys)
    """
    cdf_file = pycdf.CDF(file_path, readonly=True)
    epoch_key = keys[0]

    epoch_all = np.array([t.timestamp() for t in np.squeeze(cdf_file[epoch_key][:])],
                         dtype=np.float64)
    data_all = [np.array(cdf_file[data_key.split(":")[0]][:], dtype=np.float32)
                for data_key in keys[1:]]

    for i in range(0, len(keys[1:])):
        if ":" in keys[i + 1]:
            data_all[i] = data_all[i][:, 0:int(keys[i + 1].split(":")[1])]

    mask = np.where((epoch_all > start_time.timestamp()) & (epoch_all < stop_time.timestamp()))[0]

    if len(mask) > 0:
        dt_ts = np.squeeze(epoch_all[slice(mask[0], mask[-1] + 1)])
        data = np.stack([np.squeeze(data_all[i][slice(mask[0], mask[-1] + 1)])
                        for i in range(0, len(data_all))], axis=1)
    else:
        dt_ts = np.array([], dtype=np.float64)
        data = np.array([], dtype=np.float32)

    return dt_ts, np.squeeze(data)


def _read_netcdf_file(file_path, start_time, stop_time, keys):
    """Worker function for reading netcdf data files.

    Parameters
    ----------
    file_path : str
        path to cdf file
    start_time : datetime.datetime
        starting time
    stop_time : datetime.datetime
        end time
    keys: tuple
        cdf keys (first key is assumed to be the epoch key, all others data keys)
    """
    nc = Dataset(file_path, "r")
    epoch_key = keys[0]

    epoch_all = np.array([t / 1000 for t in nc.variables[epoch_key][...]])
    data_all = [np.array(nc[data_key][:], dtype=np.float32) for data_key in keys[1:]]

    for i in range(0, len(keys[1:])):
        if ":" in keys[i + 1]:
            data_all[i] = data_all[i][:, 0:int(keys[i + 1].split(":")[1])]

    mask = np.where((epoch_all > start_time.timestamp()) & (epoch_all < stop_time.timestamp()))[0]

    if len(mask) > 0:
        dt_ts = np.squeeze(epoch_all[slice(mask[0], mask[-1] + 1)])
        data = np.stack([np.squeeze(data_all[i][slice(mask[0], mask[-1] + 1)])
                        for i in range(0, len(data_all))], axis=1)
    else:
        dt_ts = np.array([], dtype=np.float64)
        data = np.array([], dtype=np.float32)

    return dt_ts, np.squeeze(data)


def _read_pds3_file(file_path, start_time, stop_time, columns, skip_rows, time_format,
                    time_format_data):
    """Worker function for reading pds3 data files.

    Parameters
    ----------
    file_path : str
        path to pds3 file
    start_time : datetime.datetime
        starting time
    stop_time : datetime.datetime
        end time
    columns: tuple
        colums (first column is assumed to be the epoch column, all others data columns)
    skip_rows: int
        number of rows to skip
    time_format:
        time format
    time_format_data:
        additional data for processing time information (offset for timestamp, format string etc.)
    """
    def strtots(s, fmt):
        string = s.decode("utf-8")

        # fix datetimes with 60 seconds
        if string.endswith("60.000"):
            string = "{0}59.000".format(string[:17])

            return datetime.datetime.strptime(string, fmt).timestamp() + 1
        else:
            return datetime.datetime.strptime(string, fmt).timestamp()

    if time_format == "timestamp":
        data = np.loadtxt(file_path, skiprows=skip_rows)

        epoch_all = data[:, columns[0]] + time_format_data
    elif time_format == "string":
        data = np.loadtxt(file_path, skiprows=skip_rows,
                          converters={0: lambda s: strtots(s, time_format_data)})

        epoch_all = data[:, columns[0]]
    else:
        raise NotImplementedError("time format \"{0}\" is not supported".format(time_format))

    data_all = [np.array(data[:, col], dtype=np.float32) for col in columns[1:]]

    mask = np.where((epoch_all > start_time.timestamp()) & (epoch_all < stop_time.timestamp()))[0]

    if len(mask) > 0:
        dt_ts = np.squeeze(epoch_all[slice(mask[0], mask[-1] + 1)])
        data = np.stack([np.squeeze(data_all[i][slice(mask[0], mask[-1] + 1)])
                        for i in range(0, len(data_all))], axis=1)
    else:
        dt_ts = np.array([], dtype=np.float64)
        data = np.array([], dtype=np.float32)

    return dt_ts, np.squeeze(data)


def _resolve_urls(urls):
    """Resolve url list with regex expressions.

    Parameters
    ----------
    urls : list
        urls with regex expressions

    Returns
    -------
    list
        resolved urls
    """
    url_parents = {"/".join(url.split("/")[:-1]): [] for url in urls}

    for url in urls:
        url_parents["/".join(url.split("/")[:-1])].append(url.split("/")[-1])

    urls = []

    for url_parent in url_parents:
        response = requests.get(url_parent)

        if response.ok:
            response_text = response.text
        else:
            return response.raise_for_status()

        # match all urls with regex pattern
        soup = BeautifulSoup(response_text, "html.parser")
        hrefs = [_.get("href") for _ in soup.find_all("a")]

        for url_regex in url_parents[url_parent]:
            last_file = None

            for url_child in hrefs:
                if url_child and re.match(url_regex, url_child):
                    last_file = "/".join([url_parent, url_child])

            if last_file:
                urls.append(last_file)

    return urls
