# -*- coding: utf-8 -*-

import datetime
import logging
import numpy as np
import os
import requests
import requests_ftp
import sys
import time

from spacepy import pycdf
from threading import Thread
from queue import Queue


def configure_logging(debug=False):
    """Configure built-in python logging.

    Parameters
    ----------
    debug : bool, optional
        enable DEBUG logging, by default False
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)

    if debug:
        sh.setLevel(logging.DEBUG)
    else:
        sh.setLevel(logging.INFO)

    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(sh)


def download_files(urls, folder, max_threads=50, logger=None):
    """Downloads multiple files located at the given urls and store them at the specified output
    folder.

    Parameters
    ----------
    urls : list
        url list
    folder : str
        output folder
    max_threads : int
        maximum number of simultaniously running threads, by default 50
    logger : logging.Logger, optional
        logging handler, by default None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    timer = time.time()

    q = Queue(maxsize=0)
    num_threads = min(max_threads, len(urls))

    # fill queue with required urls
    for url in list(urls):
        file_path = os.path.join(folder, url.split("/")[-1])

        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            q.put((url, file_path))

    qsize = q.qsize()

    if qsize > 0:
        for _ in range(num_threads):
            worker = Thread(target=_download_files, args=(q, folder, logger))
            worker.setDaemon(True)
            worker.start()

        # wait until all threads completed
        q.join()

        logger.info("downloaded {0} files ({1:.2f}s)".format(qsize, time.time() - timer))


def _download_files(q, folder, logger):
    """Worker function for "download_files".

    Parameters
    ----------
    q : queue.Queue
        queue
    folder : str
        output folder
    logger : logging.Logger, optional
        logging handler
    """
    while not q.empty():
        entry = q.get()

        url = entry[0]
        file_path = entry[1]

        logger.info("downloading \"{0}\"".format(url))

        try:
            with open(file_path, "wb") as fp:
                if url.startswith("http"):
                    response = requests.get(url)

                    # fix for PPI that does not return a 404 on a bad URL
                    if url.startswith("https://pds-ppi.igpp.ucla.edu") and \
                            "Content-Type" not in response.headers:
                        raise requests.HTTPError("header missing \"Content-Type\"")
                elif url.startswith("ftp"):
                    response = requests_ftp.ftp.FTPSession().retr(url)
                else:
                    response = requests.get(url)

                if response.ok:
                    fp.write(response.content)
                else:
                    return response.raise_for_status()
        except Exception as err:
            logger.error("failed to download \"{0}\" ({1})".format(url, err))
            os.remove(file_path)

        q.task_done()


def get_day_of_year(dt):
    """Get the day of year for datetime dt.

    Parameters
    ----------
    dt : t : datetime.datetime
        time

    Returns
    -------
    int
        day of the year
    """
    return (dt - datetime.datetime(dt.year, 1, 1)).days + 1


def get_time_combination(dt):
    """Generate tuple containing information about the datetime dt.

    Parameters
    ----------
    t : datetime.datetime
        time

    Returns
    -------
    np.ndarray
        datetime information
    """
    dt_yyyy = dt.year
    dt_mm = dt.month
    dt_dd = dt.day
    dt_doy = get_day_of_year(dt)

    # doy of first and last day of the month
    if dt_mm < 12:
        dt_mm_doys = (
            get_day_of_year(datetime.datetime(dt_yyyy, dt_mm, 1)),
            get_day_of_year(datetime.datetime(dt_yyyy, dt_mm + 1, 1)
                            - datetime.timedelta(days=1)
                            )
        )
    else:
        dt_mm_doys = (
            get_day_of_year(datetime.datetime(dt_yyyy, dt_mm, 1)),
            get_day_of_year(datetime.datetime(dt_yyyy + 1, 1, 1)
                            - datetime.timedelta(days=1)
                            )
        )

    dt_yy = dt_yyyy - int(dt_yyyy - dt_yyyy % 1000)

    return np.array([dt_yyyy, dt_mm, dt_dd, dt_doy, dt_mm_doys[0], dt_mm_doys[1], dt_yy])


def _read_cdf_file(file_path, start_time, stop_time, keys, stride):
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
        cdf keys
    stride : int
        data stride
    """
    cdf_file = pycdf.CDF(file_path, readonly=True)
    epoch_key = keys[0]

    epoch_all = np.array([t.timestamp() for t in np.squeeze(cdf_file[epoch_key][:])],
                         dtype=np.float64)
    data_all = [np.array(cdf_file[data_key][:], dtype=np.float32) for data_key in keys[1:]]

    mask = np.where((epoch_all > start_time.timestamp()) & (epoch_all < stop_time.timestamp()))[0]

    if len(mask) > 0:
        dt_ts = np.squeeze(epoch_all[slice(mask[0], mask[-1] + 1)][::stride])
        data = np.stack([np.squeeze(data_all[i][slice(mask[0], mask[-1] + 1)][::stride])
                        for i in range(0, len(data_all))], axis=1)
    else:
        dt_ts = np.array([], dtype=np.float64)
        data = np.array([], dtype=np.float32)

    return dt_ts, data


def _read_npy_file(file_path, start_time, stop_time, stride):
    """Worker function for reading raw npy mag data files.

    Parameters
    ----------
    file_path : str
        path to npy file
    start_time : datetime.datetime
        starting time
    stop_time : datetime.datetime
        end time
    stride : int
        data stride
    """
    npy_file = np.load(file_path)

    epoch_all = np.array(npy_file[:, 0], dtype=np.float64)
    data_all = np.array(npy_file[:, 1:], dtype=np.float32)

    mask = np.where((epoch_all > start_time.timestamp()) & (epoch_all < stop_time.timestamp()))[0]

    if len(mask) > 0:
        dt_ts = np.squeeze(epoch_all[slice(mask[0], mask[-1] + 1)][::stride])
        data = np.squeeze(data_all[slice(mask[0], mask[-1] + 1)][::stride])
    else:
        dt_ts = np.array([], dtype=np.float64)
        data = np.array([], dtype=np.float32)

    return dt_ts, data
