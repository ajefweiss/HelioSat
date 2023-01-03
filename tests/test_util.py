# -*- coding: utf-8 -*-

"""test_util.py

Unit tests for utility functions.
"""

import datetime as dt
import os

import pytest
import requests

from heliosat import util


def test_dt_utc():
    assert util.dt_utc(2020, 1, 1).tzinfo == dt.timezone.utc


def test_dt_utc_from_str():
    # test all implemented strptime formats
    test_dt = dt.datetime.now()
    for fmt in util._strptime_formats:
        test_dt_conv = util.dt_utc_from_str(test_dt.strftime(fmt))
        assert test_dt_conv.tzinfo == dt.timezone.utc
        assert test_dt_conv.year == test_dt.year
        assert test_dt_conv.month == test_dt.month
        assert test_dt_conv.day == test_dt.day

    with pytest.raises(ValueError):
        util.dt_utc_from_str("ZEBRA")

    # test bad function argument
    try:
        util.dt_utc_from_str("2022-01-01V345f")
    except ValueError:
        assert True
    except Exception:
        assert False


def test_dt_utc_from_ts():
    assert util.dt_utc_from_ts(1669737000).tzinfo == dt.timezone.utc
    assert util.dt_utc_from_ts(1669737000.0).tzinfo == dt.timezone.utc


def test_fetch_url():
    _ = util.fetch_url("https://en.wikipedia.org/wiki/Solar_physics")

    with pytest.raises(requests.HTTPError):
        util.fetch_url("https://en.wikipedia.org/wiki/Solar_physics_bad")

    # test ftp
    _ = util.fetch_url(
        "ftp://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/fk/bc_sci_v08.tf"
    )


def test_get_any():
    td = dict({"c": 1})

    assert util.get_any(td, ["a", "b", "c"]) == 1


def test_load_json():
    kfile = util.load_json("./heliosat/spacecraft/kernels.json")

    assert len(kfile["kernels"]) > 0


def test_sanitize_dt():
    dtp_a = dt.datetime(2020, 1, 1)

    dtp_list_a = [
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        dt.datetime(2020, 1, 3),
    ]
    dtp_list_b = ["2022-01-01T06:23:07", "2022-01-01T07:00:07", "2022-01-01T07:01:07"]

    assert util.sanitize_dt(dtp_a).tzinfo == dt.timezone.utc
    # missing test for proper tz conversion
    assert util.sanitize_dt("2022-01-01T06:23:07").tzinfo == dt.timezone.utc

    # test w/ list of dtp
    dtp_list_a_r = util.sanitize_dt(dtp_list_a)
    dtp_list_b_r = util.sanitize_dt(dtp_list_b)

    assert len(dtp_list_a_r) == len(dtp_list_a)
    assert len(dtp_list_b_r) == len(dtp_list_b)

    for _ in dtp_list_a_r:
        assert isinstance(_, dt.datetime)
        assert _.tzinfo == dt.timezone.utc

    for _ in dtp_list_b_r:
        assert isinstance(_, dt.datetime)
        assert _.tzinfo == dt.timezone.utc


@pytest.mark.skip(reason="wip")
def test_url_regex_files():
    raise NotImplementedError


def test_url_regex_resolve():
    test_url = "$https://www.ngdc.noaa.gov/dscovr/data/2020/01/oe_m1m_dscovr_s20200101000000_e20200101235959_p(\\d{14})_pub.nc.gz"  # noqa: E501
    test_url_2 = "$https://www.ngdc.noaa.gov/dscovr/data/2020/01/oe_m1m_dscovr_s(\\d{14})_e(\\d{14})_p(\\d{14})_pub.nc.gz"  # noqa: E501

    result, groups = util.url_regex_resolve(test_url, reduce=False)

    assert (
        os.path.basename(result[0])
        == "oe_m1m_dscovr_s20200101000000_e20200101235959_p20200102022225_pub.nc.gz"
    )
    assert groups[0] == "20200102022225"

    result_2 = util.url_regex_resolve(test_url_2, reduce=True)

    assert (
        result_2
        == "https://www.ngdc.noaa.gov/dscovr/data/2020/01/oe_m1m_dscovr_s20200131000000_e20200131235959_p20200201020023_pub.nc.gz"  # noqa: E501
    )
