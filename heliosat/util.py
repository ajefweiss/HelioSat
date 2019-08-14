# -*- coding: utf-8 -*-

import logging
import os
import requests
import requests_ftp
import sys
import time

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

        # wait until all threads are completed
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

            # remove empty file
            if os.path.isfile(file_path):
                os.remove(file_path)

        q.task_done()
