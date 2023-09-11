# -*- coding: utf-8 -*-

"""spacecraft.py
"""

import concurrent.futures
import datetime as dt
import logging as lg
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import spiceypy

import heliosat

from .caching import (
    cache_add_entry,
    cache_entry_exists,
    cache_generate_key,
    cache_get_entry,
)
from .datafile import DataFile
from .smoothing import smooth_data
from .util import dt_utc, dt_utc_from_ts, get_any, pop_any, sanitize_dt


class Body(object):
    """Body class."""

    name: str
    name_naif: str

    def __init__(
        self,
        name: str,
        name_naif: str,
        kernel_group: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.name = name
        self.name_naif = name_naif

        if "default" not in heliosat._skm.group_list:
            heliosat._skm.load_group("default")

        if kernel_group:
            heliosat._skm.load_group(kernel_group, **kwargs)

    def trajectory(
        self,
        dtp: Union[dt.datetime, Sequence[dt.datetime]],
        observer: str = "SUN",
        units: str = "AU",
        **kwargs: Any
    ) -> np.ndarray:
        dtp = sanitize_dt(dtp)

        reference_frame = get_any(kwargs, ["reference_frame", "frame"], "J2000")
        sampling_freq = get_any(
            kwargs, ["sampling_freq", "sampling_frequency", "sampling_rate"], 3600
        )

        # use dt list as endpoints
        if kwargs.pop("as_endpoints", False):
            dtp = _generate_endpoints(dtp, sampling_freq)

        traj = np.array(
            spiceypy.spkpos(
                self.name_naif,
                spiceypy.datetime2et(dtp),
                reference_frame,
                "NONE",
                observer,
            )[0]
        )

        if units == "AU":
            traj *= 6.68459e-9
        elif units == "m":
            traj *= 1e3
        elif units == "km":
            pass
        else:
            raise ValueError('unit "{0!s}" is not supported'.format(units))

        return traj


class Spacecraft(Body):
    """Spacecraft class."""

    name: str
    name_naif: str
    kernel_group: str

    _json: dict

    data_file_class = DataFile

    def __init__(self, **kwargs: Any) -> None:
        super(Spacecraft, self).__init__(
            self.name, self.name_naif, self.kernel_group, **kwargs
        )

    @property
    def data_keys(self) -> List[str]:
        return list(self._json["keys"].keys())

    def data_key_resolve(self, data_key: str) -> str:
        if data_key not in self._json["keys"]:
            resolved = False

            for key in self._json["keys"]:
                if data_key in self._json["keys"][key].get("alt_keys", []):
                    data_key = key
                    resolved = True
                    break

            if not resolved:
                raise KeyError('data_key "{0!s}" not found'.format(data_key))

        return data_key

    def get(
        self,
        dtp: Union[str, dt.datetime, Sequence[str], Sequence[dt.datetime]],
        data_key: str,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger = lg.getLogger(__name__)

        data_key = self.data_key_resolve(data_key)

        if isinstance(dtp, dt.datetime):
            dtp = [dtp]

        dtp = sanitize_dt(dtp)

        # extract cache argument before generating identifier
        cached = pop_any(kwargs, ["cached", "cache", "use_cache"], False)

        # caching identifier
        identifiers = {
            "data_key": data_key,
            "spacecraft": self.name,
            "times": [_t.timestamp() for _t in dtp],
            "version": heliosat.__version__,
            **kwargs,
        }

        # extract relevant kwargs
        remove_nans = kwargs.pop("remove_nans", False)
        return_datetimes = kwargs.pop("return_datetimes", False)
        sampling_freq = get_any(
            kwargs, ["sampling_freq", "sampling_frequency", "sampling_rate"], 60
        )

        smoothing_kwargs = {"smoothing": kwargs.pop("smoothing", "closest")}

        # get additional smoothing args
        for key in dict(kwargs):
            if "smoothing" in key:
                smoothing_kwargs[key] = kwargs.pop(key)

        if cached:
            cache_key = cache_generate_key(identifiers)

            if cache_entry_exists(cache_key):
                dtp_r, dk_r = cache_get_entry(cache_key)

                return dtp_r, dk_r
            else:
                logger.info('cache entry "%s" not found', cache_key)

        # use dt list as endpoints
        if kwargs.pop("as_endpoints", False):
            dtp = _generate_endpoints(dtp, sampling_freq)

        dtp_r, dk_r = self._get_data(dtp[0], dtp[-1], data_key, **kwargs)

        if smoothing_kwargs["smoothing"]:
            dtp_r, dk_r = smooth_data(dtp, dtp_r, dk_r, **smoothing_kwargs)

        if return_datetimes:
            dtp_r = dt_utc_from_ts(dtp_r)

        if remove_nans:
            nanfilter = np.invert(np.any(np.isnan(dk_r[:, :]), axis=1))

            if not return_datetimes:
                dtp_r = dtp_r[nanfilter]

            dk_r = dk_r[nanfilter]

        if cached:
            logger.info('generating cache entry "%s"', cache_key)
            cache_add_entry(cache_key, (dtp_r, dk_r))

        return dtp_r, dk_r

    def _get_files(
        self,
        dt_start: dt.datetime,
        dt_end: dt.datetime,
        data_key: str,
        force_download: bool = False,
        skip_download: bool = False,
    ) -> List[DataFile]:
        day = dt_start - dt.timedelta(
            hours=dt_start.hour, minutes=dt_start.minute, seconds=dt_start.second
        )

        # adjust ranges slightly
        if dt_end.hour == 0 and dt_end.minute == 0 and dt_end.second == 0:
            dt_end -= dt.timedelta(seconds=1)

        files = []

        # prepare urls
        while True:
            url = _replace_date_string(self._json["keys"][data_key]["base_url"], day)

            if self._json["keys"][data_key].get("file_expr", None):
                file_expr = _replace_date_string(
                    self._json["keys"][data_key]["file_expr"], day
                )
            else:
                file_expr = None

            files.append(self.data_file_class(url, file_expr, data_key, self._json))

            day += dt.timedelta(hours=24)

            if day > dt_end:
                break

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_prepare_file, file, force_download, skip_download)
                for file in files
            ]

            for future in concurrent.futures.as_completed(futures):
                _ = future.result()

        files_ready = []

        for file in files:
            if file.ready:
                files_ready.append(file)

        return files_ready

    def _get_data(
        self, dt_start: dt.datetime, dt_end: dt.datetime, data_key: str, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger = lg.getLogger(__name__)

        data_key = self.data_key_resolve(data_key)

        dt_start = sanitize_dt(dt_start)
        dt_end = sanitize_dt(dt_end)

        if dt_start > dt_end:
            raise ValueError("starting date must be before final date")

        force_download = kwargs.get("force_download", False)
        skip_download = kwargs.get("skip_download", False)
        transform_batch_size = kwargs.get("transform_batch_size", 0)
        skip_sort = kwargs.get("skip_sort", False)

        if self.__class__.__name__.startswith("CL"):
            print("fixing cluster ranges")
            dt_start -= dt.timedelta(days=1)
            dt_end += dt.timedelta(days=1)

        # get necessary files
        files = self._get_files(
            dt_start,
            dt_end,
            data_key,
            force_download=force_download,
            skip_download=skip_download,
        )

        if len(files) == 0:
            raise Exception("no valid files to continue with")

        logger.info(
            "using %s files to generate " "data in between %s - %s",
            len(files),
            dt_start,
            dt_end,
        )

        columns = kwargs.get("columns", ["~"])
        columns.extend(kwargs.get("extra_columns", []))
        reference_frame = get_any(kwargs, ["reference_frame", "frame"], None)

        if self.__class__.__name__.startswith("CL"):
            dt_start += dt.timedelta(days=1)
            dt_end -= dt.timedelta(days=1)

        if len(files) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        _read_file,
                        file,
                        dt_start,
                        dt_end,
                        data_key,
                        columns,
                        self.kernel_group,
                        reference_frame,
                        transform_batch_size,
                        skip_sort,
                    )
                    for file in files
                ]

            result = [_ for _ in [future.result() for future in futures] if _]

            dtp_r = np.concatenate([_[0] for _ in result])
            dk_r = np.concatenate([_[1] for _ in result])

            return dtp_r, dk_r
        else:
            result = _read_file(
                files[0],
                dt_start,
                dt_end,
                data_key,
                columns,
                self.kernel_group,
                reference_frame,
                transform_batch_size,
                skip_sort,
            )

            return result


def _generate_endpoints(
    dtp: Sequence[dt.datetime], sampling_freq: int
) -> Sequence[dt.datetime]:
    if len(dtp) < 2:
        raise ValueError("datetime list must be of length larger of 2 to use endpoints")

    n_dtp = int((dtp[-1].timestamp() - dtp[0].timestamp()) // sampling_freq)
    _ = np.linspace(dtp[0].timestamp(), dtp[-1].timestamp(), n_dtp)
    dtp = [dt.datetime.fromtimestamp(_, dt.timezone.utc) for _ in _]

    return dtp


def _prepare_file(
    file_obj: DataFile, force_download: bool = False, skip_download: bool = False
) -> None:
    logger = lg.getLogger(__name__)

    file_obj.ready = False

    if not force_download:
        if file_obj.local():
            return

    if skip_download:
        logger.error('skipping data file "%s"', os.path.basename(file_obj.base_url))
        return

    # add functionality for remote compressed files
    compression = file_obj._json["keys"][file_obj.data_key].get("compression", None)
    if compression == "gz" and not file_obj.base_url.endswith("gz"):
        file_obj.base_url = file_obj.base_url + ".gz"

    if file_obj.download():
        return

    logger.error("failed to fetch data file (%s)", os.path.basename(file_obj.base_url))


def _read_file(file_obj: DataFile, *args: Tuple[Any]) -> Tuple[np.ndarray, np.ndarray]:
    return file_obj.read(*args)


def _replace_date_string(string: str, day: dt.datetime) -> str:
    string = string.replace("{YYYY}", str(day.year))
    string = string.replace("{YY}", "{0:02d}".format(day.year % 100))
    string = string.replace("{MM}", "{:02d}".format(day.month))
    string = string.replace("{MONTH}", day.strftime("%B")[:3].upper())
    string = string.replace("{DD}", "{:02d}".format(day.day))
    string = string.replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

    doym1 = dt_utc(day.year, day.month, 1)

    if day.month == 12:
        doym2 = dt_utc(day.year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        doym2 = dt_utc(day.year, day.month + 1, 1) - dt.timedelta(days=1)

    string = string.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
    string = string.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

    return string
