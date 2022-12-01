# -*- coding: utf-8 -*-

"""util.py

Implement basic utility functions such as datetime conversions. Designed for internal use only.
"""

import datetime as dt
import json
import logging as lg
import os
import re
import requests
import requests_ftp
import sys

from bs4 import BeautifulSoup
from typing import Any, List, Optional, Sequence, Union


_strptime_formats = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d"
]


def dt_utc(*args: Any) -> dt.datetime:
    return dt.datetime(*args).replace(tzinfo=dt.timezone.utc)


def dt_utc_from_str(string: str, string_format: Optional[str] = None) -> dt.datetime:
    dtp = None

    if string_format:
        try:
            dtp = dt.datetime.strptime(string, string_format)
        except ValueError:
            for fmt in _strptime_formats:
                try:
                    dtp = dt.datetime.strptime(string, fmt)
                except ValueError:
                    pass
    else:
        for fmt in _strptime_formats:
            try:
                dtp = dt.datetime.strptime(string, fmt)
            except ValueError:
                pass

    if dtp:
        if not dtp.tzinfo:
            dtp = dtp.replace(tzinfo=dt.timezone.utc)
        return dtp

    raise ValueError("could not convert \"{0!s}\", unkown format".format(string))


def dt_utc_from_ts(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def fetch_url(url: str) -> bytes:
    logger = lg.getLogger(__name__)

    if url.startswith("http"):
        logger.debug("fetching url (http) \"%s\"", url)
        response = requests.get(url)
    elif url.startswith("ftp"):
        logger.debug("fetching url (ftp) \"%s\"", url)
        requests_ftp.monkeypatch_session()
        s = requests.Session()
        response = s.get(url)
        s.close()
    else:
        raise requests.HTTPError("invalid url \"{0!s}\"".format(url))

    if response.ok:
        # fix for url's that return 200 instead of a 404
        if int(response.headers.get("Content-Length", 0)) < 1000:
            raise requests.HTTPError("Content-Length is very small"
                                     "(url is most likely not a valid file)")

        return response.content
    else:
        raise requests.HTTPError("failed to fetch url \"{0!s}\" ({1})".format(url, response.status_code))


def get_any(kwargs: dict, keys: Sequence[str], default: Any = None) -> Any:
    while len(keys) > 0:
        if keys[0] in kwargs:
            return kwargs.get(keys[0])

        keys = keys[1:]

    return default


def load_json(path: str) -> dict:
    with open(path, "r") as fh:
        json_dict = dict(json.load(fh))

    return json_dict


def sanitize_dt(dtp: Union[str, dt.datetime, Sequence[str], Sequence[dt.datetime]]) -> Union[dt.datetime, Sequence[dt.datetime]]:
    if isinstance(dtp, dt.datetime) and dtp.tzinfo is None:
        return dtp.replace(tzinfo=dt.timezone.utc)
    elif isinstance(dtp, dt.datetime) and dtp.tzinfo != dt.timezone.utc:
        return dtp.astimezone(dt.timezone.utc)
    elif isinstance(dtp, str):
        return dt_utc_from_str(dtp)
    elif hasattr(dtp, "__iter__"):
        _dt = list(dtp)

        if isinstance(_dt[0], dt.datetime):
            for i in range(len(_dt)):
                if _dt[i].tzinfo is None:  
                    _dt[i] = _dt[i].replace(tzinfo=dt.timezone.utc)  
                else:
                    _dt[i] = _dt[i].astimezone(dt.timezone.utc)  
        elif isinstance(_dt[0], str):
            _dt = [dt_utc_from_str(_) for _ in _dt]

        return _dt  
    else:
        return dtp  


def url_regex_files(url: str, folder: str) -> List[str]:
    logger = lg.getLogger(__name__)

    local_files = os.listdir(folder)
    url_pattern = os.path.basename(url)

    matched_files = []

    for local_file in local_files:
        if re.match(url_pattern, local_file):
            matched_files.append(os.path.join(folder, local_file))
    
    return matched_files


def url_regex_resolve(url: str, reduce: bool = False) -> Union[str, List[str]]:
    url_parent = os.path.dirname(url[1:])
    url_regex = os.path.basename(url[1:])

    urls_expanded = []
    
    response = requests.get(url_parent, timeout=20)

    if response.ok:
            response_text = response.text
    else:
        raise requests.HTTPError("failed to fetch url \"{0!s}\" ({1})".format(url_parent, response.status_code))

    # match all url's with regex pattern
    soup = BeautifulSoup(response_text, "html.parser")
    hrefs = [_.get("href") for _ in soup.find_all("a")]

    for url_child in [_.get("href") for _ in soup.find_all("a")]:
        if url_child and re.match(url_regex, url_child):
            urls_expanded.append("/".join([url_parent, url_child]))

    if reduce:
        return urls_expanded[-1]
    else:
        return urls_expanded
