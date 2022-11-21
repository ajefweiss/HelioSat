# -*- coding: utf-8 -*-

"""util.py

Implement basic utility functions such as datetime conversions. Designed for internal use only.
"""

import datetime
import json
import logging
import os
import re
import requests
import requests_ftp
import sys

from bs4 import BeautifulSoup
from typing import Any, List, Optional, Sequence, Union


def dt_utc(*args: Any) -> datetime.datetime:
    return datetime.datetime(*args).replace(tzinfo=datetime.timezone.utc)


def dt_utc_from_str(string: str, string_format: Optional[str] = None) -> datetime.datetime:
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d"
    ]

    dt = None

    if string_format:
        try:
            dt = datetime.datetime.strptime(string, string_format)
        except ValueError:
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(string, fmt)
                except ValueError:
                    pass
    else:
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(string, fmt)
            except ValueError:
                pass

    if dt:
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    raise ValueError("could not convert \"{0!s}\", unkown format".format(string))


def dt_utc_from_ts(ts: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def fetch_url(url: str) -> bytes:
    logger = logging.getLogger(__name__)

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


def sanitize_dt(dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]]) -> Union[datetime.datetime, Sequence[datetime.datetime]]:
    if isinstance(dt, datetime.datetime) and dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    elif isinstance(dt, datetime.datetime):
        return dt.astimezone(datetime.timezone.utc)
    elif isinstance(dt, str):
        return dt_utc_from_str(dt)
    elif hasattr(dt, "__iter__"):
        _dt = list(dt)

        if isinstance(_dt[0], datetime.datetime):
            for i in range(len(_dt)):
                if _dt[i].tzinfo is None:  
                    _dt[i] = _dt[i].replace(tzinfo=datetime.timezone.utc)  
                else:
                    _dt[i] = _dt[i].astimezone(datetime.timezone.utc)  
        elif isinstance(_dt[0], str):
            _dt = [dt_utc_from_str(_) for _ in _dt]

        return _dt  
    else:
        return dt  


def url_regex_files(url: str, folder: str) -> List[str]:
    logger = logging.getLogger(__name__)

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
