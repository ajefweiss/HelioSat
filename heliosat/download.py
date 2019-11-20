# -*- coding: utf-8 -*-

"""download.py

Utility functions for downloading data files. Intended for internal use only.
"""

import logging
import os
import requests
import requests_ftp

from threading import Thread
from queue import Queue


def download_files(file_urls, file_paths, **kwargs):
    """Download files from given url's and store them locally.

    Parameters
    ----------
    file_urls : list
        Target url's.
    file_paths : Union[list, str]
        Destination file paths (optionally destination folder).

    Other Parameters
    ----------------
    force: bool
        Force overwrite (default is False).
    logger : logging.Logger
        Logger handle (default is None).
    threads: int
        Number of parallel threads (default is 20).

    Raises
    ------
    ValueError
        If file_paths is not a folder and the number of given file_paths does not match the number
        of given url's.
    """
    force = kwargs.get("force", kwargs.get("overwrite", False))
    logger = kwargs.get("logger", logging.getLogger(__name__))
    threads = min(len(file_urls), kwargs.get("threads", 20))

    if isinstance(file_paths, str):
        if not os.path.exists(file_paths):
            os.makedirs(file_paths)

        file_paths = [os.path.join(file_paths, file_url.split("/")[-1]) for file_url in file_urls]

    if len(file_urls) != len(file_paths):
        logger.exception("invalid file path list (size mismatch)")
        raise ValueError("invalid file path list (size mismatch)")

    queue = Queue(maxsize=0)

    for i in range(len(file_urls)):
        queue.put((file_urls[i], file_paths[i]))

    logger.debug("attempting to download %i files", queue.qsize())

    workers = []

    for _ in range(threads):
        worker = Thread(target=download_files_worker, args=(queue, force, logger))
        worker.setDaemon(True)
        worker.start()

        workers.append(worker)

    # wait until all threads are completed
    queue.join()


def download_files_worker(q, force, logger):
    """Worker function for downloading files.

    Parameters
    ----------
    q : queue.Queue
        Worker queue.
    force : bool
        Force overwrite.
    logger : logging.Logger
        Logger handle.

    Raises
    ------
    requests.HTTPError
        If download fails or file is smaller than 1000 bytes (occurs in some cases when the website
        returns 200 despite the file not existing for some sites).
    NotImplementedError
        If url is not http(s) or ftp.
    """
    while not q.empty():
        try:
            (file_url, file_path) = q.get(True, 86400)
            logger.debug("downloading \"%s\"", file_url)

            file_exists = os.path.isfile(file_path)

            if not force and file_exists:
                if file_url.startswith("http"):
                    response = requests.get(file_url, stream=True, timeout=60)
                    size = response.headers.get('Content-Length', -1)

                    # skip download if file appears to be the same (by size)
                    if os.path.getsize(file_path) == int(size):
                        continue

            with open(file_path, "wb") as file:
                if file_url.startswith("http"):
                    response = requests.get(file_url)
                elif file_url.startswith("ftp"):
                    ftp_session = requests_ftp.ftp.FTPSession()
                    response = ftp_session.retr(file_url)
                    ftp_session.close()
                else:
                    logger.exception("invalid url: \"%s\"", file_url)
                    raise NotImplementedError("invalid url: \"%s\"", file_url)

                if response.ok:
                    # fix for url's that return 200 instead of a 404
                    if "Content-Length" in response.headers and \
                            int(response.headers.get("Content-Length")) < 1000:
                        raise requests.HTTPError("Content-Length is very small"
                                                 "(url is most likely is not a valid file)")
                    else:
                        file.write(response.content)
                else:
                    return response.raise_for_status()
        except requests.HTTPError as error:
            logger.warning("failed to download \"%s\" (%s)", file_url, error)

            # remove file (only if it was created by failed download)
            if not file_exists:
                os.remove(file_path)
        finally:
            q.task_done()
