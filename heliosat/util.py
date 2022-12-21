# -*- coding: utf-8 -*-

"""util.py

Implement basic utility functions such as datetime conversions. Designed for internal use only.
"""

import datetime as dt
import json
import logging as lg
import os
import re
from typing import Any, List, Optional, Sequence, Union

import requests
import requests_ftp
from bs4 import BeautifulSoup

_strptime_formats = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
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

    raise ValueError('could not convert "{0!s}", unkown format'.format(string))


def dt_utc_from_ts(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def fetch_url(url: str, return_headers: bool = False) -> bytes:
    logger = lg.getLogger(__name__)

    if url.startswith("http"):
        logger.debug('fetching url (http) "%s"', url)
        response = requests.get(url)
    elif url.startswith("ftp"):
        logger.debug('fetching url (ftp) "%s"', url)
        requests_ftp.monkeypatch_session()
        s = requests.Session()
        response = s.get(url)
        s.close()
    else:
        raise requests.HTTPError('invalid url "{0!s}"'.format(url))

    if response.ok:
        # fix for url's that return 200 instead of a 404
        if int(response.headers.get("Content-Length", 1000)) < 1000:
            raise requests.HTTPError(
                "Content-Length is very small l=%i" "(url is most likely not a valid file)",
                int(response.headers.get("Content-Length", 0)),
            )

        if return_headers:
            return response.content, response.headers
        else:
            return response.content
    else:
        raise requests.HTTPError('failed to fetch url "{0!s}" ({1})'.format(url, response.status_code))


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


def sanitize_dt(
    dtp: Union[str, dt.datetime, Sequence[str], Sequence[dt.datetime]]
) -> Union[dt.datetime, Sequence[dt.datetime]]:
    if isinstance(dtp, dt.datetime) and dtp.tzinfo is None:
        return dtp.replace(tzinfo=dt.timezone.utc)
    elif isinstance(dtp, dt.datetime) and dtp.tzinfo != dt.timezone.utc:
        return dtp.astimezone(dt.timezone.utc)
    elif isinstance(dtp, str):
        return dt_utc_from_str(dtp)
    elif hasattr(dtp, "__iter__"):
        _dtp = list(dtp)

        if isinstance(_dtp[0], dt.datetime):
            for i in range(len(_dtp)):
                if _dtp[i].tzinfo is None:
                    _dtp[i] = _dtp[i].replace(tzinfo=dt.timezone.utc)
                else:
                    _dtp[i] = _dtp[i].astimezone(dt.timezone.utc)
        elif isinstance(_dtp[0], str):
            _dtp = [dt_utc_from_str(_) for _ in _dtp]

        return _dtp
    else:
        return dtp


def url_regex_files(url: str, folder: str) -> List[str]:
    local_files = os.listdir(folder)
    url_pattern = os.path.basename(url)

    matched_files = []
    matched_groups = []

    for local_file in local_files:
        match = re.match(url_pattern, local_file)

        if match:
            matched_files.append(os.path.join(folder, local_file))

            groups = match.groups()

            if len(groups) >= 1:
                matched_groups.append(groups[-1])

    return matched_files, matched_groups


def url_regex_resolve(url: str, reduce: bool = False) -> Union[str, List[str]]:
    url_parent = os.path.dirname(url[1:])
    url_regex = os.path.basename(url[1:])

    urls_expanded = []
    urls_groups = []

    response = requests.get(url_parent, timeout=20)

    if response.ok:
        response_text = response.text
    else:
        raise requests.HTTPError('failed to fetch url "{0!s}" ({1})'.format(url_parent, response.status_code))

    # match all url's with regex pattern
    soup = BeautifulSoup(response_text, "html.parser")

    for url_child in [_.get("href") for _ in soup.find_all("a")]:
        match = re.match(url_regex, url_child)

        if url_child and match:
            urls_expanded.append("/".join([url_parent, url_child]))

            groups = match.groups()

            if len(groups) >= 1:
                urls_groups.append(groups[-1])

    if reduce:
        return urls_expanded[-1]
    else:
        return urls_expanded, urls_groups
