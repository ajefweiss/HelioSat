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
import time

from bs4 import BeautifulSoup
from netCDF4 import Dataset

from heliosat.proc import smoothing_average_kernel, smoothing_adaptive_gaussian
from heliosat.util import download_files, get_time_combination, _read_cdf_file, _read_npy_file


class FaradayCup(object):
    def get_fc(self, start_time, stop_time, stride=1, ignore_missing_files=False):
        """Get raw concatenated plasma data in between start and stop time.

        Parameters
        ----------
        start_time : datetime.datetime
            start time
        stop_time : datetime.datetime
            stop time
        stride : int, optional
            data stride, by default 1
        ignore_missing_files : bool, optional
            ignore missing files, by default False

        Returns
        -------
        np.ndarray, np.ndarray
            datetime timestamps, plasma data
        """
        logger = logging.getLogger(__name__)

        # sanity checks
        assert start_time < stop_time, "start_time must be before stop_time"
        assert start_time > self._start_time, "start_time must be after mission start time"

        if hasattr(self, "_stop_time"):
            assert stop_time < self._stop_time, "stop_time must be before mission stop time"

        fc_files = self.get_fc_files(start_time, stop_time)

        if ignore_missing_files:
            # remove missing files
            for fc_file in list(fc_files):
                if not os.path.exists(fc_file):
                    fc_files.remove(fc_file)

            if len(fc_files) == 0:
                raise FileNotFoundError("no raw data files found")
        else:
            for fc_file in list(fc_files):
                if not os.path.exists(fc_file):
                    raise FileNotFoundError("raw data file \"{0}\" "
                                            "could not be found".format(fc_file))

        logger.info("using {0} files to generating plasma "
                    "data in between {1} - {2}".format(len(fc_files), start_time, stop_time))

        processes = min(multiprocessing.cpu_count() * 2, len(fc_files))
        pool = multiprocessing.Pool(processes=processes)

        if self._fc_file_format == "cdf":
            results = pool.starmap(_read_cdf_file,
                                   [(fc_files[i], start_time, stop_time, self._fc_cdf_keys,
                                     stride) for i in range(0, len(fc_files))])
        elif self._fc_file_format == "npy":
            results = pool.starmap(_read_npy_file, [(fc_files[i], start_time, stop_time, stride)
                                   for i in range(0, len(fc_files))])

        # total length of records
        tlength = [len(result[0]) for result in results]

        dt_ts = np.empty(sum(tlength), dtype=np.float64)
        fc_data = np.empty((sum(tlength), 3), dtype=np.float32)

        for i in range(0, len(results)):
            if len(results[i][0]) > 0:
                dt_ts[sum(tlength[:i]):sum(tlength[:i + 1])] = results[i][0]
                fc_data[sum(tlength[:i]):sum(tlength[:i + 1])] = results[i][1]

        # remove NaN values
        if hasattr(self, "_fc_valid_range"):
            valid_index = np.where(np.array(fc_data[:, 0] > self._mag_valid_range[0]) &
                                   np.array(fc_data[:, 0] < self._mag_valid_range[1]))

            dt_ts = dt_ts[valid_index]
            fc_data = fc_data[valid_index]

        return dt_ts, fc_data

    def get_fc_proc(self, t, smoothing=None, smoothing_scale=None, raw_data=None):
        """Evaluate plasma at times t.

        Parameters
        ----------
        t : Iterable[datetime.datetime]
            observer time
        smoothing : str, optional
            smoothing type, by default None
        smoothing_scale : float, optional
            smoothing scale (seconds), by default None
        raw_data : (np.ndarray, np.ndarray)), optional
            raw data from get_fc, by default None
        """
        logger = logging.getLogger(__name__)

        timer = time.time()

        if raw_data is None:
            if smoothing_scale:
                # add buffer for smoothing
                dt_ts, fc_data = self.get_fc(
                    t[0] - datetime.timedelta(seconds=10 * smoothing_scale),
                    t[-1] + datetime.timedelta(seconds=10 * smoothing_scale),
                    stride=1, ignore_missing_files=False
                    )
            else:
                dt_ts, fc_data = self.get_fc(t[0], t[-1], stride=1, ignore_missing_files=False)
        else:
            dt_ts, fc_data = raw_data

            # sanity checks
            assert all([_t > datetime.datetime.fromtimestamp(dt_ts[0]) for _t in t])
            assert all([_t < datetime.datetime.fromtimestamp(dt_ts[-1]) for _t in t])

        fc_data_proc = np.zeros((len(t), 3), dtype=np.float32)

        # apply smoothing
        if smoothing == "average" and smoothing_scale:
            smoothing_average_kernel(np.array([_t.timestamp() for _t in t]), dt_ts, fc_data,
                                     fc_data_proc, float(smoothing_scale))
            logger.info("smoothing with average kernel, scale={0}s "
                        "({1:.1f}s)".format(smoothing_scale, time.time() - timer))
        elif smoothing == "adaptive_gaussian" and smoothing_scale:
            smoothing_adaptive_gaussian(np.array([_t.timestamp() for _t in t]), dt_ts, fc_data,
                                        fc_data_proc, float(smoothing_scale))
            logger.info("smoothing with adaptive gaussian method, scale={0}s "
                        "({1:.1f}s)".format(smoothing_scale, time.time() - timer))
        else:
            raise NotImplementedError("smoothing method {0} is not implemented".format(smoothing))

        return fc_data_proc


def get_fc_files_dscovr(obj, start_time, stop_time):
    """Get raw plasma data files for the DSCOVR spacecraft. The files are downloaded if
    necessary.

    Parameters
    ----------
    obj: FaradayCup
        faraday cup instance
    start_time : datetime.datetime
        start time
    stop_time : datetime.datetime
        stop time

    Returns
    -------
    list
        raw plasma data files
    """
    logger = logging.getLogger(__name__)

    # get year, month, day combinations for [tstart, tend]
    t_comb = [get_time_combination(start_time)]

    while start_time < stop_time:
        start_time += datetime.timedelta(days=1)

        t_comb.append(get_time_combination(start_time))

    # build urls
    urls = [obj._fc_remote_folder + obj._fc_remote_files.format(*list(entry[:3]))
            for entry in t_comb]

    for url in list(urls):
        file_name = url.split("/")[-1].split(".")[0] + ".npy"

        if os.path.exists(os.path.join(obj._data_folder, file_name)):
            logger.info("found {0}, skipping download".format(file_name))
            urls.remove(url)

    # get files with processing time
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

        # match all urls"s with regex pattern
        soup = BeautifulSoup(response_text, "html.parser")
        hrefs = [_.get("href") for _ in soup.find_all("a")]

        for url_regex in url_parents[url_parent]:
            last_file = None

            for url_child in hrefs:
                if url_child and re.match(url_regex, url_child):
                    last_file = "/".join([url_parent, url_child])

            if last_file:
                urls.append(last_file)

    file_urls = list(urls)

    for url in list(urls):
        file_name = url.split("/")[-1].split(".")[0] + ".npy"

        if os.path.exists(os.path.join(obj._data_folder, file_name)):
            logger.info("found {0}, skipping download".format(file_name))
            urls.remove(url)

    download_files(urls, obj._data_folder, logger=logger)

    # convert all .nc files to .npy
    if len(urls) > 0:
        logger.info("converting {0} dscovr data files".format(len(urls)))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2)

        pool.starmap(_convert_dscovr_to_npy,
                     [(os.path.join(obj._data_folder, urls[i].split("/")[-1]),)
                      for i in range(0, len(urls))])

    files = [os.path.join(obj._data_folder, url.split("/")[-1].split(".")[0] + ".npy")
             for url in file_urls]

    return files


def get_fc_files_stereo(self, start_time, stop_time):
    """Get raw plasma data files for the DSCOVR spacecraft. The files are downloaded if
    necessary.

    Parameters
    ----------
    start_time : datetime.datetime
        start time
    stop_time : datetime.datetime
        stop time

    Returns
    -------
    list
        raw plasma data files
    """
    logger = logging.getLogger(__name__)

    # get year, month, day combinations for [tstart, tend]
    t_comb = [get_time_combination(start_time)]

    while start_time < stop_time:
        start_time += datetime.timedelta(days=1)

        t_comb.append(get_time_combination(start_time))

    def st_plastic_ver(t):
        if datetime.datetime(t[0], t[1], t[2]) < datetime.datetime(2007, 1, 1):
            return "V08"
        elif datetime.datetime(t[0], t[1], t[2]) < datetime.datetime(2011, 6, 1):
            return "V09"
        else:
            return "V10"

    # build urls
    urls = [self._fc_remote_folder +
            self._fc_remote_files.format(*list(entry[:3]), st_plastic_ver(entry))
            for entry in t_comb]

    download_files(urls, self._data_folder, logger=logger)

    files = [
        os.path.join(self._data_folder, urls[i].split("/")[-1])
        for i in range(0, len(urls))
    ]

    return files


def _convert_dscovr_to_npy(file_path):
    """Worker functions for converting dscovr data files to numpy array format.

    Parameters
    ----------
    file_path : str
        path to data file
    """
    logger = logging.getLogger(__name__)

    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).split(".")[0]

    try:
        with gzip.open(file_path, "rb") as file_gz:
            with open(".".join(file_path.split(".")[:-1]), "wb") as file_nc:
                shutil.copyfileobj(file_gz, file_nc)

        nc = Dataset(".".join(file_path.split(".")[:-1]), "r")

        print(nc.variables)

        data_t = np.array([t / 1000 for t in nc.variables["time"][...]])
        data_rho = np.array(nc.variables["proton_density"][...], dtype=np.float32)
        data_v = np.array(nc.variables["proton_speed"][...], dtype=np.float32)
        data_k = np.array(nc.variables["proton_temperature"][...], dtype=np.float32)

        data = np.vstack((data_t, data_rho, data_v, data_k))

        nc.close()

        np.save(os.path.join(file_dir, file_name) + ".npy", data.T)
        os.remove(file_path)

        if os.path.exists(os.path.join(file_dir, file_name) + ".nc"):
            os.remove(os.path.join(file_dir, file_name) + ".nc")

    except Exception as err:
        logger.error("failed to convert dscovr data file (\"{0}\")".format(err))

        if os.path.exists(os.path.join(file_dir, file_name) + ".nc"):
            os.remove(os.path.join(file_dir, file_name) + ".nc")

        if os.path.exists(os.path.join(file_dir, file_name) + ".npy"):
            os.remove(os.path.join(file_dir, file_name) + ".npy")
