# -*- coding: utf-8 -*-

"""download.py

Utility functions for downloading data files. Intended for internal use only.
"""

import logging
import numpy as np
import os
import requests
import requests_ftp

from concurrent.futures import ThreadPoolExecutor


def download_files(file_urls, file_paths, **kwargs):
    """Download files from given url's and store them locally.

    Parameters
    ----------
    file_urls : list[str]
        Target url's.
    file_paths : Union[list[str], str]
        Destination file paths (or optionally the destination folder).

    Other Parameters
    ----------------
    force: bool
        Force overwrite, by default False.
    logger : logging.Logger
        Logger handle, by default None.
    threads: int
        Number of parallel threads, by default 32.

    Returns
    -------
    list[bool]
        Flags if files were downloaded succesfully, or they already existed.

    Raises
    ------
    ValueError
        If file_paths is not a folder and the number of given file_paths does not match the number
        of given url's.
    """
    force = kwargs.get("force", kwargs.get("overwrite", False))
    logger = kwargs.get("logger", logging.getLogger(__name__))
    threads = min(len(file_urls), kwargs.get("threads", 32))

    if isinstance(file_paths, str):
        if not os.path.exists(file_paths):
            os.makedirs(file_paths)

        file_paths = [os.path.join(file_paths, file_url.split("/")[-1]) for file_url in file_urls]

    if len(file_urls) != len(file_paths):
        logger.exception("invalid file path list (length mismatch)")
        raise ValueError("invalid file path list (length mismatch)")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = executor.map(worker_download_files, [(file_urls[i], file_paths[i], force, logger)
                                                       for i in range(len(file_urls))])

    results = [_ for _ in futures]

    if np.any(np.invert(results)):
        logger.warning("checked/downloaded %i/%i files", np.sum(results), len(results))
    else:
        logger.info("checked/downloaded %i files", len(results))

    return results


def worker_download_files(args):
    """Worker function for downloading files.

    Parameters
    ----------
    args: (str, str, bool, logging.Logger)
        Function arguments as tuple.

    Returns
    -------
    bool
        Flag if file was downloaded, or already existed.

    Raises
    ------
    NotImplementedError
        If url is not http(s) or ftp.
    """
    try:
        (file_url, file_path, force, logger) = args

        file_already_exists = os.path.isfile(file_path)

        if file_already_exists and os.path.getsize(file_path) == 0:
            file_already_exists = False

        if not force and file_already_exists:
            if file_url.startswith("http"):
                response = requests.get(file_url, stream=True, timeout=20)
                size = response.headers.get('Content-Length', -1)

                # skip download if file appears to be the same (checking size only)
                if os.path.getsize(file_path) == int(size):
                    logger.debug("checked \"%s\"", file_url)
                    return True

        logger.debug("downloading \"%s\"", file_url)

        if file_url.startswith("http"):
            response = requests.get(file_url)
        elif file_url.startswith("ftp"):
            ftp_session = requests_ftp.ftp.FTPSession()
            response = ftp_session.retr(file_url)
            ftp_session.close()
        else:
            logger.exception("invalid url \"%s\"", file_url)
            raise NotImplementedError("invalid url \"%s\"", file_url)

        if response.ok:
            # fix for url's that return 200 instead of a 404
            if "Content-Length" in response.headers and \
                    int(response.headers.get("Content-Length")) < 1000:
                raise requests.HTTPError("Content-Length is very small"
                                         "(url is most likely not a valid file)")
            else:
                with open(file_path, "wb") as file:
                    file.write(response.content)
        else:
            return response.raise_for_status()

        return True
    except requests.RequestException as error:
        if file_already_exists:
            logger.error("failed to check existing file \"%s\" (%s)", file_url, error)

            # local file is assumed to be correct if it cannot be checked
            return True
        else:
            logger.error("failed to download \"%s\" (%s)", file_url, error)

            # delete empty file that may have been created
            if os.path.exists(file_path):
                os.remove(file_path)

        return False
