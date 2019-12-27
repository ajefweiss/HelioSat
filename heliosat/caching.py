# -*- coding: utf-8 -*-

"""caching.py

Implements simple caching functionality.

All cached files are stored in a single folder located at either ~/.heliosat/cache or
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


def generate_cache_key(identifiers):
    """Generate SHA256 digest from the dictionary which acts as key for cached data.

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


def delete_cache_entry(key):
    """Delete cache entry belonging to specified key.

    Parameters
    ----------
    key : str
        Cache key.

    Raises
    ------
    KeyError
        If no entry for the given key is found.
    """
    logger = logging.getLogger(__name__)

    cache_path = heliosat._paths["cache"]

    if not cache_entry_exists(key):
        logger.exception("cache key \"%s\" does not exist", key)
        raise KeyError("cache key \"%s\" does not exist", key)

    os.remove(os.path.join(cache_path, "{0}.cache".format(key)))


def get_cache_entry(key):
    """Retrieve cache entry specified by key.

    Parameters
    ----------
    key : str
        Cache key.

    Returns
    -------
    str
        Data object.

    Raises
    ------
    KeyError
        If no entry for the given key is found.
    """
    logger = logging.getLogger(__name__)

    cache_path = heliosat._paths["cache"]

    if not cache_entry_exists(key):
        logger.exception("cache key \"%s\" does not exist", key)
        raise KeyError("cache key \"%s\" does not exist", key)

    with open(os.path.join(cache_path, "{0}.cache".format(key)), "rb") as pickle_file:
        cache_data = pickle.load(pickle_file)

    return cache_data


def cache_entry_exists(key):
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


def set_cache_entry(key, obj):
    """Store data in cache using the specified key.

    Parameters
    ----------
    key : str
        Cache key.
    obj : object
        Data object.
    """
    cache_path = heliosat._paths["cache"]

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    with open(os.path.join(cache_path, "{}.cache".format(key)), "wb") as pickle_file:
        pickle.dump(obj, pickle_file)
