# -*- coding: utf-8 -*-

"""util.py

General utility functions for the HelioSat package. For internal use only.
"""

import datetime
import logging
import os
import re
import requests
import requests_ftp

from bs4 import BeautifulSoup


def datetime_to_string(time_datetime):
    """Convert python datetime to string.

    Parameters
    ----------
    time_datetime : datetime.datetime
        Datetime object.

    Returns
    -------
    str
        Datetime string.
    """
    return time_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")


def get_heliosat_paths():
    """Get heliosat file paths.
    """
    root_path = os.getenv('HELIOSAT_DATAPATH', os.path.join(os.path.expanduser("~"), ".heliosat"))

    return {
        "cache": os.path.join(root_path, "cache"),
        "data": os.path.join(root_path, "data"),
        "helcats": os.path.join(root_path, "helcats"),
        "kernels": os.path.join(root_path, "kernels"),
        "root": root_path
    }


def string_to_datetime(time_string):
    """Convert string to python datetime.

    Parameters
    ----------
    time_string : str
        Datetime string.

    Returns
    -------
    datetime.datetime
        Datetime object.
    """
    try:
        return datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%MZ")


def urls_expand(urls, pre="$"):
    """"Expand list of url's using regular expressions. Both HTTP and FTP url's are supported.

    Parameters
    ----------
    urls : list
        Url list (including expandables and non-expandables).
    pre : str
        Prefix to identify expandable url's.

    Returns
    -------
    list
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


def urls_resolve(urls):
    """Resolve list of url's using regular expressions.

    Parameters
    ----------
    urls : list
        Url list.

    Returns
    -------
    list
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
