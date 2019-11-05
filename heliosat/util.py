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
    return datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")


def urls_build(fmt, range_start, range_end, versions):
    """Build url list from format string and range.

    Parameters
    ----------
    fmt : str
        Format string.
    range_start : datetime.datetime
        Time range start datetime.
    range_end : datetime.datetime
        Time range end datetime.
    versions : list
        Version list.

    Returns
    -------
    list
        Url list.

    Raises
    ------
    RuntimeError
        If no version information for a specific date is found in spacecraft.json.
    """
    logger = logging.getLogger(__name__)

    urls = []

    range_start -= datetime.timedelta(hours=range_start.hour, minutes=range_start.minute,
                                      seconds=range_start.second)

    # if range_end is the start of a new day, move it slightly back
    if range_end.hour == 0 and range_end.minute == 0 and range_end.second == 0:
        range_end -= datetime.timedelta(seconds=1)

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

        if day.month == 12:
            doym2 = datetime.datetime(day.year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            doym2 = datetime.datetime(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

        url = url.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
        url = url.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

        if versions:
            version_found = False

            for version in versions:
                if string_to_datetime(version["version_start"]) <= \
                   day < string_to_datetime(version["version_end"]):
                    version_found = True

                    for i in range(len(version["identifiers"])):
                        url = url.replace("{{V{0}}}".format(i), version["identifiers"][i])

                    break

            if not version_found:
                logger.exception("no version found for %s", day)
                raise RuntimeError("no version found for %s", day)

        urls.append(url)

    return urls


def urls_expand(urls):
    """"Expand list of url's using regular expressions. Both HTTP and FTP url's are supported.
    Only url's with a "$" prefix are expanded.

    Parameters
    ----------
    urls : list
        Url list (including expandables and non-expandables).

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
