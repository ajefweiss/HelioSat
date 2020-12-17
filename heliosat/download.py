# -*- coding: utf-8 -*-

"""download.py

Utility functions for downloading data files. Intended for internal use only.
"""

import logging
import numpy as np
import os
import re
import requests
import requests_ftp


from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union


def download_files(file_urls: List[str], file_paths: Union[str, List[str]], **kwargs) -> List[bool]:
    """Download files from given url's and store them locally.

    Parameters
    ----------
    file_urls : List[str]
        Target url's.
    file_paths : Union[str, list[str]]
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
    List[bool]
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
        futures = executor.map(_worker_download_files, [(file_urls[i], file_paths[i], force, logger)
                                                        for i in range(len(file_urls))])

    results = [_ for _ in futures]

    if np.any(np.invert(results)):
        logger.warning("checked/downloaded %i/%i files", np.sum(results), len(results))
    else:
        logger.info("checked/downloaded %i files", len(results))

    return results


def _worker_download_files(args: tuple) -> bool:
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
            # skip download if file already exists
            logger.info("skipped \"%s\"", file_url)
            return True

        if file_url.startswith("http"):
            logger.debug("downloading \"%s\"", file_url)
            response = requests.get(file_url)
        elif file_url.startswith("ftp"):
            logger.debug("downloading (ftp) \"%s\"", file_url)
            requests_ftp.monkeypatch_session()
            s = requests.Session()
            response = s.get(file_url)
            s.close()
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


def urls_expand(urls: List[str], pre: str = "$") -> list:
    """"Expand list of url's using regular expressions. Both HTTP and FTP url's are supported.

    Parameters
    ----------
    urls : List[str]
        Url list (including expandables and non-expandables).
    pre : str
        Prefix to identify expandable url's.

    Returns
    -------
    List[str]
        Expanded url list (including non-expandables).

    Raises
    ------
    NotImplementedError
        If url is not http(s) or ftp.
    """
    logger = logging.getLogger(__name__)

    # copy url's that do not need expansion
    urls_expanded = [_ for _ in urls if not _.startswith("$")]

    for url in [_ for _ in urls if _.startswith(pre)]:
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
            raise NotImplementedError("ftp is currently not support for expansion")
        else:
            logger.exception("failed to expand url: \"%s\"", url)
            raise NotImplementedError("failed to expand url: \"%s\"", url)

    return urls_expanded


def urls_resolve(urls: List[str]) -> List[str]:
    """Resolve list of url's using regular expressions.

    Parameters
    ----------
    urls : List[str]
        Url list.

    Returns
    -------
    List[str]
        Resolved url list.
    """
    # organize url's so that any page is only called once
    url_parents = {"/".join(url.split("/")[:-1]): [] for url in urls}

    for url in urls:
        url_parents["/".join(url.split("/")[:-1])].append(url.split("/")[-1])

    urls_resolved = []

    for url_parent in url_parents:
        response = requests.get(url_parent)

        if response.ok:
            response_text = response.text
        else:
            return response.raise_for_status()

        # match all url's with regex pattern
        soup = BeautifulSoup(response_text, "html.parser")
        hrefs = [_.get("href") for _ in soup.find_all("a")]

        for url_regex in url_parents[url_parent]:
            # incase of multiple matches, choose the last one
            last_match = None

            for url_child in hrefs:
                if url_child and re.match(url_regex, url_child):
                    last_match = "/".join([url_parent, url_child])

            if last_match:
                urls_resolved.append(last_match)

    return urls_resolved
