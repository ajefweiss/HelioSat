# -*- coding: utf-8 -*-

"""util.py

Utility functions for the HelioSat package. For internal use only.
"""

import datetime
import logging
import os
import re
import requests
import requests_ftp

from bs4 import BeautifulSoup
from threading import Thread
from queue import Queue


def download_files(file_urls, file_paths, **kwargs):
    """Download files from given urls and store them locally.

    Parameters
    ----------
    file_urls : list
        target urls
    file_paths : Union[list, str]
        destination file paths (or optionally destination folder)

    Raises
    ------
    ValueError
        if file_paths is not a folder and the number of given file_paths does not match the number
        of given urls
    """
    force = kwargs.get("force",  kwargs.get("overwrite", False))
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
    """Worker function for downloading files from given urls and storing them locally.

    Parameters
    ----------
    q : queue.Queue
        worker queue
    force : bool
        force overwrite
    logger : logging.Logger
        logger handle

    Raises
    ------
    requests.HTTPError
        if download fails or file is smaller than 1000 bytes (occurs when the website returns 200
        despite the file not existing for some sites)
    NotImplementedError
        if url is not in http(s) or ftp format
    """
    while not q.empty():
        try:
            (file_url, file_path) = q.get(True, 86400)
            logger.debug("downloading \"%s\"", file_url)

            if not force and os.path.isfile(file_path):
                if file_url.startswith("http"):
                    size = int(requests.get(file_url, stream=True).headers['Content-length'])

                    # skip download if file appears to be the same (by size)
                    if os.path.getsize(file_path) == size:
                        continue

            with open(file_path, "wb") as file:
                if file_url.startswith("http"):
                    response = requests.get(file_url)

                    # fix for urls that return 200 instead of a 404
                    if int(response.headers["Content-Length"]) < 1000:
                        raise requests.HTTPError("Content-Length is very small"
                                                 "(url most likely is not a valid file)")
                elif file_url.startswith("ftp"):
                    ftp_session = requests_ftp.ftp.FTPSession()
                    response = ftp_session.retr(file_url)
                    ftp_session.close()
                else:
                    logger.exception("invalid url: \"%s\"", file_url)
                    raise NotImplementedError("invalid url: \"%s\"", file_url)

                if response.ok:
                    file.write(response.content)
                else:
                    return response.raise_for_status()
        except requests.HTTPError as error:
            logger.error("failed to download \"%s\" (%s)", file_url, error)

            # remove file (empty only)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                os.remove(file_path)
        finally:
            q.task_done()


def expand_urls(urls):
    """"Expand list of url's using regex. Both HTTP and FTP url's are supported.
    Only url's with a "$" prefix are expanded.

    Parameters
    ----------
    urls : list
        url list (including expandables and non-expandables)

    Returns
    -------
    list
        expanded url list (including non-expandables)

    Raises
    ------
    NotImplementedError
        if url is not in http(s) or ftp format
    """
    logger = logging.getLogger(__name__)

    urls_expanded = [_ for _ in urls if not _.startswith("$")]

    for url in [_ for _ in urls if _.startswith("$")]:
        if url[1:].startswith("http"):
            url_parent = "/".join(url[1:].split("/")[:-1])
            url_regex = url.split("/")[-1]

            response = requests.get(url_parent)

            if response.ok:
                response_text = response.text
            else:
                return response.raise_for_status()

            # match all urls's with regex pattern
            soup = BeautifulSoup(response_text, "html.parser")

            for url_child in [_.get("href") for _ in soup.find_all("a")]:
                if url_child and re.match(url_regex, url_child):
                    urls_expanded.append("/".join([url_parent, url_child]))
        elif url[1:].startswith("ftp"):
            url_parent = "/".join(url[1:].split("/")[:-1])
            url_regex = url.split("/")[-1]

            response = requests_ftp.ftp.FTPSession().list(url_parent)

            if response.ok:
                response_text = response.text
            else:
                return response.raise_for_status()

            # match all urls's with regex pattern
            filenames = [line.split()[-1] for line in response.content.decode("utf-8").splitlines()]

            for filename in filenames:
                if re.match(url_regex, filename):
                    urls_expanded.append("/".join([url_parent, filename]))
        else:
            logger.exception("invalid url: \"%s\"", url)
            raise NotImplementedError("invalid url: \"%s\"", url)

    return urls_expanded


def get_paths():
    """Get heliosat file paths.
    """
    root_path = os.getenv('HELIOSAT_DATAPATH', os.path.join(os.path.expanduser("~"), ".heliosat"))

    return {
        "cache": os.path.join(root_path, "cache"),
        "data": os.path.join(root_path, "data"),
        "kernels": os.path.join(root_path, "kernels"),
        "root": root_path
    }


def strptime(tstr):
    """Convert string to datetime (UTC).

    Parameters
    ----------
    tstr : [type]
        string in ISO 8601 format

    Returns
    -------
    datetime.datetime
        datetime object
    """
    return datetime.datetime.strptime(tstr, "%Y-%m-%dT%H:%M:%S.%f")


def strftime(dt):
    """Convert datetime to string (UTC).

    Parameters
    ----------
    dt : datetime.datetime
        datetime object
    Returns
    -------
    str
        string in ISO 8601 format
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


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
