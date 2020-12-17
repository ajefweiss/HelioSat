# -*- coding: utf-8 -*-

"""util.py

General utility functions for the HelioSat package. For internal use only.
"""

import datetime
import logging
import os


from typing import Iterable, List, Optional, Union


def datetime_utc(*args: tuple) -> datetime.datetime:
    """Create tz-aware datetime object (UTC).
    """
    return datetime.datetime(*args).replace(tzinfo=datetime.timezone.utc)


def datetime_utc_timestamp(timestamp: float) -> datetime.datetime:
    """Create tz-aware datetime object (UTC) from timestamp.
    """
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
        "cache": os.path.join(root_path, "cache"),
        "data": os.path.join(root_path, "data"),
        "kernels": os.path.join(root_path, "kernels"),
        "root": root_path
    }


def sanitize_datetimes(t: Union[datetime.datetime, Iterable[datetime.datetime]]) \
        -> Union[datetime.datetime, Iterable[datetime.datetime]]:
    """Sanitize tz-unaware datetime objects. Any tz-unaware datetime objects are set to UTC.

    Parameters
    ----------
    t : Union[datetime.datetime, Iterable[datetime.datetime]]
        Datetime objects to sanitize.

    Returns
    -------
    Union[datetime.datetime, Iterable[datetime.datetime]]
        Sainitized datetime objects.
    """
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
    """Convert string to python datetime object.

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

    Raises
    ------
    ValueError
        If string format is unkown.
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
