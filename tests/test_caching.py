# -*- coding: utf-8 -*-

"""test_caching.py

Unit tests for caching functionality.
"""

import os
import pickle

import pytest

from heliosat import caching


def test_cache_all():
    test_cache_key = caching.cache_generate_key(dict({"prop": True}))

    assert (
        test_cache_key
        == "52c3fee076fd00dc6e6e5ecfea7b19a5be15215ce955ca6f201ce527dba348fa"
    )

    cache_path = caching.cache_add_entry(test_cache_key, dict({"TEST_KEY": 1}))

    assert os.path.exists(caching.cache_get_path())
    assert isinstance(cache_path, str)
    assert caching.cache_entry_exists(test_cache_key)
    assert os.path.exists(cache_path)

    with open(cache_path, "rb") as pickle_file:
        cache_data = pickle.load(pickle_file)

    assert cache_data["TEST_KEY"] == 1
    assert caching.cache_get_entry(test_cache_key)["TEST_KEY"] == 1

    caching.cache_delete_entry(test_cache_key)

    assert not caching.cache_entry_exists(test_cache_key)
    assert not os.path.exists(cache_path)

    # check invalid key exception
    with pytest.raises(KeyError):
        caching.cache_get_entry("INVALID_KEY")
