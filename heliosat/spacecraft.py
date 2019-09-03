# -*- coding: utf-8 -*-

"""spacecraft.py

Implements the spacecraft base class and any specific spacecraft.
"""

import datetime
import gzip
import heliosat
import json
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

from heliosat.caching import gen_key, get_cache, has_cache, set_cache
from heliosat.smoothing import smooth_data
from heliosat.spice import SpiceObject, spice_load, transform_frame
from heliosat.util import download_files, strptime


class Spacecraft(SpiceObject):
    """Base class for spacecraft.
    """
    spacecraft = None

    def __init__(self, name, **kwargs):
        logger = logging.getLogger(__name__)

        super(Spacecraft, self).__init__(name, **kwargs)

        if heliosat._spice.get("spacecraft", None) is None:
            logger.info("loading available spacecraft")

            module_path = os.path.dirname(heliosat.__file__)

            with open(os.path.join(module_path, "json/spacecraft.json")) as json_file:
                heliosat._spice["spacecraft"] = json.load(json_file)

            self.spacecraft = heliosat._spice["spacecraft"][name]
        elif name in heliosat._spice["spacecraft"]:
            self.spacecraft = heliosat._spice["spacecraft"][name]
        else:
            raise NotImplementedError("spacecraft \"%s\" is not implemented", name)

        if kwargs.get("kernel_group", None) is None and \
           self.spacecraft.get("kernel_group", None):
            spice_load(self.spacecraft["kernel_group"])

    def get_data(self, t, data_type, **kwargs):
        """Get processed data for type "data_type" at specified times t.

        Parameters
        ----------
        t : list[datetime.datetime]
            times
        data_type : str
            data type

        Returns
        -------
        np.ndarray
            processed data array

        Raises
        ------
        KeyError
            if no reference frame for the data is specified
        """
        cache_dict = {
                "data_type": data_type,
                "spacecraft": self.name,
                "times": [_t.timestamp() for _t in t]
                }

        cache_dict.update(kwargs)

        cache = kwargs.pop("cache", False)
        frame = kwargs.get("frame", None)
        smoothing = kwargs.get("smoothing", None)

        smoothing_dict = {"smoothing": smoothing}

        for key in dict(kwargs):
            if "smoothing" in key:
                smoothing_dict[key] = kwargs.pop(key)

        if cache:
            cache_key = gen_key(cache_dict)

            if has_cache(cache_key):
                return get_cache(cache_key)

        time, data = self.get_data_raw(t[0], t[-1], data_type, **kwargs)

        if smoothing:
            time, data = smooth_data(t, time, data, **smoothing_dict)

        if frame and "frame" in self.spacecraft["data"][data_type]:
            data = transform_frame(time, data, self.spacecraft["data"][data_type]["frame"], frame)
        elif frame and not self.spacecraft["data"][data_type].get("frame"):
            raise KeyError("no reference frame defined for data_type \"%s\"", data_type)

        if cache and not has_cache(cache_key):
            set_cache(cache_key, (time, data))

        return time, data

    def get_data_raw(self, range_start, range_end, data_type, **kwargs):
        """Get raw data for type "data_type" within specified time range.

        Parameters
        ----------
        range_start : datetime.datetime
            time range start
        range_end : datetime.datetime
            time range end
        data_type : str
            data type

        Returns
        -------
        np.ndarray
            raw data array
        """
        logger = logging.getLogger(__name__)

        raw_files = self.get_data_files(range_start, range_end, data_type)

        logger.info("using %s files to generate "
                    "data in between %s - %s", len(raw_files), range_start, range_end)

        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count() * 2, len(raw_files)))

        kwargs_read = dict(self.spacecraft["data"][data_type])

        if "extra_columns" in kwargs:
            kwargs_read["columns"].extend(kwargs["extra_columns"])

        results = pool.starmap(read_file, [(raw_files[i], range_start, range_end,
                                            kwargs_read)
                                           for i in range(len(raw_files))])

        # concatenate data
        lengths = [len(result[0]) for result in results]
        columns = len(results[0][1][0])

        time = np.empty(sum(lengths), dtype=np.float64)
        data = np.empty((sum(lengths), columns), dtype=np.float32)

        for i in range(0, len(results)):
            if len(results[i][0]) > 0:
                time[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][0]
                data[sum(lengths[:i]):sum(lengths[:i + 1])] = results[i][1]

        return time, data

    def get_data_files(self, range_start, range_end, data_type):
        """Get raw data file paths for type "data_type" within specified time range.

        Parameters
        ----------
        range_start : datetime.datetime
            time range start
        range_end : datetime.datetime
            time range end
        data_type : str
            data type

        Returns
        -------
        list
            raw data file paths

        Raises
        ------
        ValueError
            if invalid time range is given
        """
        logger = logging.getLogger(__name__)

        if range_start < self.mission_start or range_end > self.mission_end:
            raise ValueError("invalid time range (must be within %s - %s)",
                             strptime(self.mission_start),
                             strptime(self.mission_end))

        data_path = os.path.join(heliosat._paths["data"], data_type)

        urls = urls_build(self.spacecraft["data"][data_type]["urls"], range_start, range_end,
                          versions=self.spacecraft["data"][data_type].get("versions", None))

        # resolve regex urls if required
        if self.spacecraft["data"][data_type].get("use_regex", False):
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

        return files

    @property
    def data_types(self):
        return list(self.spacecraft["data"].keys())

    @property
    def mission_start(self):
        return strptime(self.spacecraft["mission_start"])

    @property
    def mission_end(self):
        if "mission_end" in self.spacecraft:
            return strptime(self.spacecraft.get("mission_end"))
        else:
            return datetime.datetime.now()


class DSCOVR(Spacecraft):
    def __init__(self):
        super(DSCOVR, self).__init__("dscovr", body_name="EARTH")


class MES(Spacecraft):
    def __init__(self):
        super(MES, self).__init__("messenger", body_name="MESSENGER")


class PSP(Spacecraft):
    def __init__(self):
        from spiceypy import boddef, bodn2c
        from spiceypy.utils.support_types import SpiceyError

        super(PSP, self).__init__("psp", body_name="SPP")

        try:
            bodn2c("SPP_SPACECRAFT")
        except SpiceyError:
            # kernel fix
            boddef("SPP_SPACECRAFT", bodn2c("SPP"))


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
        file path
    range_start : datetime.datetime
        time range start
    range_end : datetime.datetime
        time range end
    kwargs : dict
        additional parameters

    Returns
    -------
    (np.ndarray, np.ndarray)
        time and data array

    Raises
    ------
    NotImplementedError
        if data format is not supported
    NotImplementedError
        if time format is not supported (pds3 only)
    """
    columns = kwargs.get("columns")
    format = kwargs.get("format")
    values = kwargs.get("values", None)

    if format == "cdf":
        file = pycdf.CDF(file_path, readonly=True)

        time = np.array([t.timestamp() for t in np.squeeze(file[columns[0]][:])], dtype=np.float64)
        data = [np.array(file[key.split(":")[0]][:], dtype=np.float32) for key in columns[1:]]
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
            string = string.decode("utf-8")

            # fix datetimes with 60 seconds
            if string.endswith("60.000"):
                # TODO: fix :17 (only works for one specific format)
                string = "{0}59.000".format(string[:17])

                return datetime.datetime.strptime(string, format).timestamp() + 1
            else:
                return datetime.datetime.strptime(string, format).timestamp()

        if time_format_type == "string":
            file = np.loadtxt(file_path, skiprows=skip_rows,
                              converters={0: lambda s: decode_string(s, time_format)})

            time = file[:, columns[0]]
        elif time_format_type == "timestamp":
            file = np.loadtxt(file_path, skiprows=skip_rows)

            time = file[:, columns[0]] + time_offset
        else:
            raise NotImplementedError("time_format_type \"%s\" is not implemented",
                                      time_format_type)

        data = [np.array(file[:, key], dtype=np.float32) for key in columns[1:]]
    else:
        raise NotImplementedError("format \"%s\" is not implemented", format)

    # fix some odd shapes
    for i in range(len(data)):
        if data[i].ndim == 1:
            data[i] = data[i].reshape(-1, 1)

    # discard unneeded dimensions
    # TODO: add extra commands
    if isinstance(columns[0], str):
        for i in range(0, len(columns[1:])):
            if ":" in columns[i + 1]:
                data[i] = data[i][:, 0:int(columns[i + 1].split(":")[1])]

    mask = np.where((time > range_start.timestamp()) & (time < range_end.timestamp()))[0]

    if len(mask) > 0:
        mask_slice = slice(mask[0], mask[-1] + 1)
        time_part = np.squeeze(time[mask_slice])

        for i in range(len(data)):
            if i == 0:
                data_part = data[0][mask_slice]
            else:
                data_part = np.hstack((data_part, data[i][mask_slice]))

        data_part = np.squeeze(data_part)

        # remove invalid values
        if values:
            valid_indices = np.where(
                np.array(data_part[:, 0] > values[0]) &
                np.array(data_part[:, 0] < values[1])
                )

            time_part = time_part[valid_indices]
            data_part = data_part[valid_indices]
    else:
        time_part = np.array([], dtype=np.float64)
        data_part = np.array([], dtype=np.float32)

    return time_part, data_part


def urls_build(fmt, range_start, range_end, versions):
    """Build url list from format string and range.

    Parameters
    ----------
    fmt : str
        format string
    range_start : datetime.datetime
        time range start
    range_end : datetime.datetime
        time range end
    versions : list
        version information

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        if no version information for a specific date is found in spacecraft.json
    """
    urls = []

    # build url for each day in range
    for day in [range_start + datetime.timedelta(days=i)
                for i in range((range_end - range_start).days + 1)]:
        url = fmt
        url = url.replace("{YYYY}", str(day.year))
        url = url.replace("{YY}", "{0:02d}".format(day.year % 100))
        url = url.replace("{MM}", "{:02d}".format(day.month))
        url = url.replace("{MONTH}", day.strftime("%B")[:3].upper())
        url = url.replace("{DD}", "{:02d}".format(day.day))
        url = url.replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

        doym1 = datetime.datetime(day.year, day.month, 1)
        doym2 = datetime.datetime(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

        url = url.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
        url = url.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

        if versions:
            version_found = False

            for version in versions:
                if strptime(version["version_start"]) <= day < strptime(version["version_end"]):
                    version_found = True

                    for i in range(len(version["identifiers"])):
                        url = url.replace("{{V{0}}}".format(i), version["identifiers"][i])

                    break

            if not version_found:
                raise RuntimeError("no version found for %s", day)

        urls.append(url)

    return urls


def urls_resolve(urls):
    """Resolve urls in list with regex expressions.

    Parameters
    ----------
    urls : list
        url list

    Returns
    -------
    list
        resolved url list
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
