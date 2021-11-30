# -*- coding: utf-8 -*-

"""spacecraft.py
"""

import concurrent.futures
import datetime
import heliosat
import logging
import multiprocessing
import numpy as np
import spiceypy

from .caching import cache_add_entry, cache_entry_exists, cache_generate_key, cache_get_entry
from .datafile import DataFile
from .smoothing import smooth_data
from .util import dt_utc, dt_utc_from_ts, sanitize_dt
from typing import Any, List, Optional, Sequence, Tuple, Union


class Body(object):
    """Body class.
    """
    name: str
    name_naif: str

    def __init__(self, name: str, name_naif: str, kernel_group: Optional[str] = None, **kwargs: Any) -> None:
        self.name = name
        self.name_naif = name_naif

        if "default" not in heliosat._skm.group_list:
            heliosat._skm.load_group("default")

        if kernel_group:
            heliosat._skm.load_group(kernel_group, **kwargs)

    def trajectory(self, dt: Union[datetime.datetime, Sequence[datetime.datetime]],
                   reference_frame: str = "J2000", observer: str = "SUN", units: str = "AU") -> np.ndarray:
        logger = logging.getLogger(__name__)

        dt = sanitize_dt(dt)

        traj = np.array(
            spiceypy.spkpos(
                self.name_naif,
                spiceypy.datetime2et(dt),
                reference_frame,
                "NONE",
                observer
            )[0]
        )

        if units == "AU":
            traj *= 6.68459e-9
        elif units == "m":
            traj *= 1e3
        elif units == "km":
            pass
        else:
            logger.exception("unit \"%s\" is not supported", units)
            raise ValueError("unit \"{0!s}\" is not supported".format(units))

        return traj


class Spacecraft(Body):
    """Spacecraft class.
    """
    name: str
    name_naif: str
    kernel_group: str

    _json: dict

    data_file_class = DataFile

    def __init__(self, **kwargs: Any) -> None:
        logger = logging.getLogger(__name__)

        super(Spacecraft, self).__init__(self.name, self.name_naif, self.kernel_group, **kwargs)

        # legacy support
        self.get_data = self.get

    def get(self, dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], data_key: str, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger(__name__)

        data_key = self.data_key_resolve(data_key)

        if isinstance(dt, datetime.datetime):
            dt = [dt]

        dt = sanitize_dt(dt)  # type: ignore

        # caching identifier
        identifiers = {
            "data_key": data_key,
            "spacecraft": self.name,
            "times": [_t.timestamp() for _t in dt],  # type: ignore
            "version": heliosat.__version__,
            **kwargs
        }

        # extract relevant kwargs
        remove_nans = kwargs.pop("remove_nans", False)
        return_datetimes = kwargs.pop("return_datetimes", False)
        sampling_freq = kwargs.pop("sampling_freq", 60)

        smoothing_kwargs = {"smoothing": kwargs.pop("smoothing", "closest")}

        # get additional smoothing args
        for key in dict(kwargs):
                if "smoothing" in key:
                    smoothing_kwargs[key] = kwargs.pop(key)

        use_cache = kwargs.pop("use_cache", False)

        if use_cache:
            cache_key = cache_generate_key(identifiers)

            if cache_entry_exists(cache_key):
                dt_r, dk_r = cache_get_entry(cache_key)

                return dt_r, dk_r
            else:
                logger.info("cache entry \"%s\" not found", cache_key)

        # use dt list as endpoints 
        if kwargs.pop("as_endpoints", False):
            if len(dt) < 2:  # type: ignore
                logger.exception("datetime list must be of length larger of 2 to use endpoints")
                raise ValueError("datetime list must be of length larger of 2 to use endpoints")

            _ = np.linspace(dt[0].timestamp(), dt[-1].timestamp(), int((dt[-1].timestamp() - dt[0].timestamp()) // sampling_freq))  # type: ignore
            dt = [datetime.datetime.fromtimestamp(_, datetime.timezone.utc) for _ in _]

        dt_r, dk_r = self._get_data(dt[0], dt[-1], data_key, **kwargs)  # type: ignore

        if smoothing_kwargs["smoothing"]:
            dt_r, dk_r = smooth_data(dt, dt_r, dk_r, **smoothing_kwargs)  # type: ignore

        if return_datetimes:
            _dt = list(dt_r)

            for i in range(len(_dt)):
                if _dt[i] != np.nan:
                    _dt[i] = dt_utc_from_ts(dt_r[i])

            dt_r = np.array(_dt)

        if remove_nans:
            nanfilter = np.invert(np.any(np.isnan(dk_r[:, :]), axis=1))

            dt_r = dt_r[nanfilter]
            dk_r = dk_r[nanfilter]

        if use_cache:
            logger.info("generating cache entry \"%s\"", cache_key)
            cache_add_entry(cache_key, (dt_r, dk_r))

        return dt_r, dk_r

    def _get_data(self, dt_start: datetime.datetime, dt_end: datetime.datetime, data_key: str, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger(__name__)

        data_key = self.data_key_resolve(data_key)

        dt_start = sanitize_dt(dt_start)  # type: ignore
        dt_end = sanitize_dt(dt_end)  # type: ignore

        if dt_start > dt_end:
            logger.exception("starting date must be before final date")
            raise ValueError("starting date must be before final date")

        force_download = kwargs.get("force_download", False)

        # get necessary files
        files = self._get_files(dt_start, dt_end, data_key, force_download=force_download)

        logger.info("using %s files to generate "
                    "data in between %s - %s", len(files), dt_start, dt_end)

        columns = kwargs.get("columns", ["~"])
        columns.extend(kwargs.get("extra_columns", []))
        frame = kwargs.get("frame", kwargs.get("reference_frame", None))

        max_workers = min([multiprocessing.cpu_count(), len(files)])

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(file.read, dt_start, dt_end, data_key, columns, frame) for file in files]

        result = [_ for _ in [future.result() for future in futures] if _]

        dt_r = np.concatenate([_[0] for _ in result])
        dk_r = np.concatenate([_[1] for _ in result])

        return dt_r, dk_r

    def _get_files(self, dt_start: datetime.datetime, dt_end: datetime.datetime, data_key: str, force_download: bool = False) -> List[DataFile]:
        logger = logging.getLogger(__name__)
        
        # adjust ranges slightly
        if (dt_end - dt_start).days > 1:
            dt_start -= datetime.timedelta(hours=dt_start.hour, minutes=dt_start.minute,
                                            seconds=dt_start.second)
            if dt_end.hour == 0 and dt_end.minute == 0 and dt_end.second == 0:
                dt_end -= datetime.timedelta(seconds=1)

        files = []

        # prepare urls
        for day in [dt_start + datetime.timedelta(days=i) for i in range((dt_end - dt_start).days + 1)]:
            base_urls = []

            for url in self._json["keys"][data_key]["base_urls"]:
                url = url.replace("{YYYY}", str(day.year))
                url = url.replace("{YY}", "{0:02d}".format(day.year % 100))
                url = url.replace("{MM}", "{:02d}".format(day.month))
                url = url.replace("{MONTH}", day.strftime("%B")[:3].upper())
                url = url.replace("{DD}", "{:02d}".format(day.day))
                url = url.replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

                doym1 = dt_utc(day.year, day.month, 1)

                if day.month == 12:
                    doym2 = dt_utc(day.year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    doym2 = dt_utc(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

                url = url.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
                url = url.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

                base_urls.append(url)

            filename = self._json["keys"][data_key].get("filename", None)

            if filename:
                filename = filename.replace("{YYYY}", str(day.year))
                filename = filename.replace("{YY}", "{0:02d}".format(day.year % 100))
                filename = filename.replace("{MM}", "{:02d}".format(day.month))
                filename = filename.replace("{MONTH}", day.strftime("%B")[:3].upper())
                filename = filename.replace("{DD}", "{:02d}".format(day.day))

                filename = filename.replace("{DOY}", "{:03d}".format(day.timetuple().tm_yday))

                doym1 = dt_utc(day.year, day.month, 1)

                if day.month == 12:
                    doym2 = dt_utc(day.year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    doym2 = dt_utc(day.year, day.month + 1, 1) - datetime.timedelta(days=1)

                filename = filename.replace("{DOYM1}", "{:03d}".format(doym1.timetuple().tm_yday))
                filename = filename.replace("{DOYM2}", "{:03d}".format(doym2.timetuple().tm_yday))

            files.append(self.data_file_class(base_urls, filename, data_key, self._json))

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(file.prepare, force_download) for file in files]

            for future in concurrent.futures.as_completed(futures):
                _ = future.result()

        for file in list(files):
            if not file.ready:
                files.remove(file)

        return files

    @property
    def data_keys(self) -> List[str]:
        return list(self._json["keys"].keys())

    def data_key_resolve(self, data_key: str) -> str:
        logger = logging.getLogger(__name__)

        if data_key not in self._json["keys"]:
            resolved = False

            for key in self._json["keys"]:
                if data_key in self._json["keys"][key].get("alt_keys", []):
                    data_key = key
                    resolved = True
                    break

            if not resolved:
                logger.exception("data_key \"%s\" not found", data_key)
                raise KeyError("data_key \"{0!s}\" not found".format(data_key))
            
        return data_key
