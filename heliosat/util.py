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
from typing import Iterable, List, Optional, Union


def datetime_utc(*args: tuple) -> datetime.datetime:
    return datetime.datetime(*args).replace(tzinfo=datetime.timezone.utc)


def datetime_utc_timestamp(timestamp: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)


def datetime_to_string(time_datetime: datetime.datetime) -> str:
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


def get_heliosat_paths() -> dict:
    """Get heliosat file paths.

    Returns
    -------
    dict
        Dictionary containing heliosat paths.
    """
    root_path = os.getenv('HELIOSAT_DATAPATH', os.path.join(os.path.expanduser("~"), ".heliosat"))

    return {
        "att_data": os.path.join(root_path, "att_data"),
        "cache": os.path.join(root_path, "cache"),
        "data": os.path.join(root_path, "data"),
        "helcats": os.path.join(root_path, "helcats"),
        "kernels": os.path.join(root_path, "kernels"),
        "root": root_path
    }


def sanitize_datetimes(t: Union[datetime.datetime, Iterable[datetime.datetime]]) \
        -> Union[datetime.datetime, Iterable[datetime.datetime]]:
    if isinstance(t, datetime.datetime) and t.tzinfo is None:
        return t.replace(tzinfo=datetime.timezone.utc)
    elif isinstance(t, datetime.datetime):
        return t.astimezone(datetime.timezone.utc)
    elif hasattr(t, "__iter__"):
        _t = list(t)

        for i in range(len(_t)):
            if _t[i].tzinfo is None:
                _t[i] = _t[i].replace(tzinfo=datetime.timezone.utc)
            else:
                _t[i] = _t[i].astimezone(datetime.timezone.utc)

        return _t
    else:
        return t


def string_to_datetime(time_string: str, time_format: Optional[str] = None) -> datetime.datetime:
    """Convert string to python datetime.

    Parameters
    ----------
    time_string : str
        Datetime string.
    time_format : str
        Datetime string format.

    Returns
    -------
    datetime.datetime
        Datetime object.
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d"
    ]

    if time_format:
        try:
            return datetime.datetime.strptime(
                time_string, time_format
                ).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            pass

    for f in formats:
        try:
            return datetime.datetime.strptime(time_string, f).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            pass

    raise ValueError("could not convert \"%s\", unkown format", time_string)


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
