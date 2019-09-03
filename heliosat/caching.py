# -*- coding: utf-8 -*-

"""smoothing.py

Implements simple caching functionality.
"""

import hashlib
import json
import os
import pickle

import heliosat


def gen_key(dictionary):
    """Generate SHA256 digest from dictionary json dump which acts as key for cached data.

    Parameters
    ----------
    dictionary : dict
         dictionary containing data identifiers

    Returns
    -------
    str
        cache key (SHA256 digest)
    """
    hashobj = hashlib.sha256()
    hashobj.update(json.dumps(dictionary, sort_keys=True).encode("utf-8"))
    return hashobj.hexdigest()


def del_cache(key):
    """Delete cached data belonging to specified key.

    Parameters
    ----------
    key : str
        cache key

    Raises
    ------
    KeyError
        if no entry for the given key is found
    """
    cache_path = heliosat._paths["cache"]

    if not has_cache(key):
        raise KeyError("cache key \"%s\" does not exist", key)

    os.remove(os.path.join(cache_path, "{0}.cache".format(key)))


def get_cache(key):
    """Retrieve stored data specified by key.

    Parameters
    ----------
    key : str
        cache key

    Returns
    -------
    str
        data object

    Raises
    ------
    KeyError
        if no entry for the given key is found
    """
    cache_path = heliosat._paths["cache"]

    if not has_cache(key):
        raise KeyError("cache key \"%s\" does not exist", key)

    with open(os.path.join(cache_path, "{0}.cache".format(key)), "rb") as pickle_file:
        cache_data = pickle.load(pickle_file)

    return cache_data


def has_cache(key):
    """Check if an entry for the given cache key exists.

    Parameters
    ----------
    key : str
        cache key

    Returns
    -------
    bool
        if key exists
    """
    cache_path = heliosat._paths["cache"]

    return os.path.exists(os.path.join(cache_path, "{}.cache".format(key)))


def set_cache(key, obj):
    """Store data in cache using specified key.

    Parameters
    ----------
    key : str
        cache key
    obj : object
        data object
    """
    cache_path = heliosat._paths["cache"]

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    with open(os.path.join(cache_path, "{}.cache".format(key)), "wb") as pickle_file:
        pickle.dump(obj, pickle_file)
