# -*- coding: utf-8 -*-

"""caching.py

Implements simple caching functionality. Designed for internal use only.

Notes: All cached files are stored in a single folder located at either ~/.heliosat/cache or
$HELIOSAT_DATAPATH/cache if the environment variable is set. The required disk space can grow to be
very large if used extensively. The folder can be deleted manually if required without the loss
of any functionality.
"""

import hashlib
import json
import logging as lg
import os
import pickle
from typing import Any


def cache_add_entry(key: str, obj: object) -> None:
    logger = lg.getLogger(__name__)

    cache_path = cache_get_path()

    if not os.path.exists(cache_path):
        logger.debug('cache path does not exist, creating "%s"', cache_path)
        os.makedirs(cache_path)

    file_path = os.path.join(cache_path, "{}.cache".format(key))

    with open(file_path, "wb") as pickle_file:
        logger.debug('creating cache entry "%s"', key)
        pickle.dump(obj, pickle_file)

    return file_path


def cache_delete_entry(key: str) -> None:
    logger = lg.getLogger(__name__)

    if not cache_entry_exists(key):
        raise KeyError('cache entry "{0!s}" does not exist'.format(key))

    logger.debug('deleting cache entry "%s"', key)
    os.remove(os.path.join(cache_get_path(), "{0}.cache".format(key)))


def cache_entry_exists(key: str) -> bool:
    return os.path.exists(os.path.join(cache_get_path(), "{}.cache".format(key)))


def cache_generate_key(identifiers: dict) -> str:
    hashobj = hashlib.sha256()
    hashobj.update(json.dumps(identifiers, sort_keys=True).encode("utf-8"))

    return hashobj.hexdigest()


def cache_get_entry(key: str) -> Any:
    logger = lg.getLogger(__name__)

    if not cache_entry_exists(key):
        raise KeyError('cache key "{0!s}" does not exist'.format(key))

    with open(os.path.join(cache_get_path(), "{0}.cache".format(key)), "rb") as pickle_file:
        logger.debug('loading cache entry "%s"', key)
        cache_data = pickle.load(pickle_file)

    return cache_data


def cache_get_path() -> str:
    return os.path.join(
        os.getenv("HELIOSAT_DATAPATH", os.path.join(os.path.expanduser("~"), ".heliosat")),
        "cache",
    )
