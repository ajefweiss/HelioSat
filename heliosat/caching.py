# -*- coding: utf-8 -*-

"""caching.py

Implements simple caching functionality. For internal use only.

Notes: All cached files are stored in a single folder located at either ~/.heliosat/cache or
$HELIOSAT_DATAPATH/cache if the environment variable is set. The required disk space can grow to be
very large if used extensively, but the folder can be deleted manually if required without the loss
of any functionality.
"""

import hashlib
import heliosat
import json
import logging
import os
import pickle


def cache_add_entry(key: str, obj: object):
    """Store data in cache using the specified key.

    Parameters
    ----------
    key : str
        Cache key.
    obj : object
        Data object.
    """
    logger = logging.getLogger(__name__)

    cache_path = heliosat._paths["cache"]

    if not os.path.exists(cache_path):
        logger.debug("cache path does not exist, creating \"%s\"", cache_path)
        os.makedirs(cache_path)

    with open(os.path.join(cache_path, "{}.cache".format(key)), "wb") as pickle_file:
        logger.debug("creating cache entry \"%s\"", key)
        pickle.dump(obj, pickle_file)


def cache_delete_entry(key: str):
    """Delete cache entry belonging to specified cache key.

    Parameters
    ----------
    key : str
        Cache key.

    Raises
    ------
    KeyError
        No entry for the given key is found.
    """
    logger = logging.getLogger(__name__)

    cache_path = heliosat._paths["cache"]

    if not cache_entry_exists(key):
        logger.exception("cache entry \"%s\" does not exist", key)
        raise KeyError("cache entry \"%s\" does not exist", key)

    logger.debug("deleting cache entry \"%s\"", key)
    os.remove(os.path.join(cache_path, "{0}.cache".format(key)))


def cache_entry_exists(key: str) -> bool:
    """Check if an entry for the given cache key exists.

    Parameters
    ----------
    key : str
        Cache key.

    Returns
    -------
    bool
        Flag if key exists.
    """
    cache_path = heliosat._paths["cache"]

    return os.path.exists(os.path.join(cache_path, "{}.cache".format(key)))


def cache_generate_key(identifiers: dict) -> str:
    """Generate SHA256 digest from the identifiers dictionary which acts as the cache key.

    Parameters
    ----------
    identifiers : dict
        Dictionary containing cache identifiers.

    Returns
    -------
    str
        Cache key (SHA256 digest).
    """
    hashobj = hashlib.sha256()
    hashobj.update(json.dumps(identifiers, sort_keys=True).encode("utf-8"))

    return hashobj.hexdigest()


def cache_get_entry(key: str) -> object:
    """Retrieve cache entry specified by cache key.

    Parameters
    ----------
    key : str
        Cache key.

    Returns
    -------
    object
        Data object.

    Raises
    ------
    KeyError
        No entry is found.
    """
    logger = logging.getLogger(__name__)

    cache_path = heliosat._paths["cache"]

    if not cache_entry_exists(key):
        logger.exception("cache key \"%s\" does not exist", key)
        raise KeyError("cache key \"%s\" does not exist", key)

    with open(os.path.join(cache_path, "{0}.cache".format(key)), "rb") as pickle_file:
        logger.debug("loading cache entry \"%s\"", key)
        cache_data = pickle.load(pickle_file)

    return cache_data
