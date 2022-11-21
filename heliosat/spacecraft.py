# -*- coding: utf-8 -*-

"""spacecraft.py
"""

import concurrent.futures
import datetime
import heliosat
import logging
import multiprocessing
import numpy as np
import os
import shutil
import spiceypy

from .caching import cache_add_entry, cache_entry_exists, cache_generate_key, cache_get_entry
from .datafile import DataFile
from .smoothing import smooth_data
from .util import dt_utc, dt_utc_from_str, dt_utc_from_ts, fetch_url, get_any, sanitize_dt, url_regex_files, url_regex_resolve
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

    def trajectory(self, dt: Union[datetime.datetime, Sequence[datetime.datetime]], observer: str = "SUN",
                   units: str = "AU", **kwargs: Any) -> np.ndarray:
        logger = logging.getLogger(__name__)

        dt = sanitize_dt(dt)

        reference_frame = get_any(kwargs, ["reference_frame", "frame"], "J2000")

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

        dt = sanitize_dt(dt)  

        # caching identifier
        identifiers = {
            "data_key": data_key,
            "spacecraft": self.name,
            "times": [_t.timestamp() for _t in dt],  
            "version": heliosat.__version__,
            **kwargs
        }

        # extract (pop) relevant kwargs
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
            if len(dt) < 2:
                raise ValueError("datetime list must be of length larger of 2 to use endpoints")

            _ = np.linspace(dt[0].timestamp(), dt[-1].timestamp(), int((dt[-1].timestamp() - dt[0].timestamp()) // sampling_freq))  
            dt = [datetime.datetime.fromtimestamp(_, datetime.timezone.utc) for _ in _]

        dt_r, dk_r = self._get_data(dt[0], dt[-1], data_key, **kwargs)

        if smoothing_kwargs["smoothing"]:
            dt_r, dk_r = smooth_data(dt, dt_r, dk_r, **smoothing_kwargs)

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

        dt_start = sanitize_dt(dt_start)  
        dt_end = sanitize_dt(dt_end)  

        if dt_start > dt_end:
            raise ValueError("starting date must be before final date")

        force_download = kwargs.get("force_download", False)
        skip_download = kwargs.get("skip_download", False)

        # get necessary files
        files = self._get_files(dt_start, dt_end, data_key, force_download=force_download, skip_download=skip_download)

        if len(files) == 0:
            raise Exception("no valid files to continue with")

        logger.info("using %s files to generate "
                    "data in between %s - %s", len(files), dt_start, dt_end)

        columns = kwargs.get("columns", ["~"])
        columns.extend(kwargs.get("extra_columns", []))
        reference_frame = get_any(kwargs, ["reference_frame", "frame"], None)

        max_workers = min([multiprocessing.cpu_count(), len(files)])

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(file.read, dt_start, dt_end, data_key, columns, self.kernel_group, reference_frame) for file in files]

        result = [_ for _ in [future.result() for future in futures] if _]

        dt_r = np.concatenate([_[0] for _ in result])
        dk_r = np.concatenate([_[1] for _ in result])

        return dt_r, dk_r

    def _get_files(self, dt_start: datetime.datetime, dt_end: datetime.datetime, data_key: str,
                   force_download: bool = False, skip_download: bool = False) -> List[DataFile]:
        # adjust ranges slightly
        if (dt_end - dt_start).days > 1:
            dt_start -= datetime.timedelta(hours=dt_start.hour, minutes=dt_start.minute,
                                            seconds=dt_start.second)
            if dt_end.hour == 0 and dt_end.minute == 0 and dt_end.second == 0:
                dt_end -= datetime.timedelta(seconds=1)

        files = []

        # prepare urls
        for day in [dt_start + datetime.timedelta(days=i) for i in range((dt_end - dt_start).days + 1)]:
            url = self._json["keys"][data_key]["base_url"]

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

            filename = self._json["keys"][data_key].get("filename", os.path.basename(self._json["keys"][data_key].get("base_url")))

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

            files.append(self.data_file_class(url, filename, data_key, self._json))

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(prepare_file, file, force_download, skip_download) for file in files]

            for future in concurrent.futures.as_completed(futures):
                _ = future.result()

        files_ready = []

        for file in files:
            if file.ready:
                files_ready.append(file)

        return files_ready

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
                raise KeyError("data_key \"{0!s}\" not found".format(data_key))
            
        return data_key


def prepare_file(file_obj, force_download: bool = False, skip_download: bool = False) -> None:
    logger = logging.getLogger(__name__)

    _version_list = list(file_obj._json["keys"][file_obj.data_key]["version_list"])
    _version_list.remove(file_obj._json["keys"][file_obj.data_key]["version_default"])

    version_list = [file_obj._json["keys"][file_obj.data_key]["version_default"]] + _version_list

    file_obj.ready = False

    # check each version for local file
    for version in version_list:
        url = file_obj.base_url.replace("{VER}", version)

        if file_obj.filename:
            filename = file_obj.filename.replace("{VER}", version)
        else:
            filename = None  

        try:
            if url.startswith("$"):
                # determine if any versions exist locally
                if filename:
                    local_files = url_regex_files(filename, file_obj.key_path)
                else:
                    local_files = url_regex_files(url, file_obj.key_path)

                if len(local_files) > 0 and not force_download:
                    file_obj.file_path = local_files[-1]
                    file_obj.version = version
                    file_obj.ready = True

                    return
            else:
                # determine if any versions exist locally
                if filename:
                     file_obj.file_path = os.path.join(file_obj.key_path, filename)
                else:
                    file_obj.file_path = os.path.join(file_obj.key_path, os.path.basename(file_obj.base_url))

                if os.path.isfile(file_obj.file_path) and os.path.getsize(file_obj.file_path) > 0:  
                    file_obj.version = version
                    file_obj.ready = True

                    return
        except Exception as e:
            logger.warning("exception when checking local file %s (%s)", file_obj.base_url, e)
            continue

    if skip_download:
        logger.error("skipping data file \"%s\"", os.path.basename(file_obj.base_url))
        return

    # add functionality for remote compressed files
    if file_obj._json["keys"][file_obj.data_key].get("compression", None) == "gz" and not file_obj.base_url.endswith("gz"):
        file_obj.base_url = file_obj.base_url + ".gz"

    # check each version for remote file
    _url_pre = None

    for version in version_list:
        url = file_obj.base_url.replace("{VER}", version)

        # skip if url does not change with version
        if _url_pre and url == _url_pre:
            continue

        _url_pre = url

        if file_obj.filename:
            filename = file_obj.filename.replace("{VER}", version)
            file_obj.file_path = os.path.join(file_obj.key_path, filename)
        else:
            file_obj.file_path = os.path.join(file_obj.key_path, os.path.basename(url))  

        try:
            if url.startswith("$"):
                url = str(url_regex_resolve(url, reduce=True))

                logger.info("fetch \"%s\"", url)
                file_data = fetch_url(url)

                file_obj.file_path = os.path.join(file_obj.key_path, os.path.basename(url)) 

                with open(file_obj.file_path, "wb") as fh:
                    fh.write(file_data)

                file_obj.file_url = url
                file_obj.version = version
                file_obj.ready = True

                # decompress
                if file_obj._json["keys"][file_obj.data_key].get("compression", None) == "gz":
                    with gzip.open(file_obj.file_path, "rb") as file_gz:
                        with open(".".join(file_obj.file_path.split(".")[:-1]), "wb") as file_extracted:
                            shutil.copyfileobj(file_gz, file_extracted)

                    os.remove(file_obj.file_path)

                    file_obj.file_path = ".".join(file_obj.file_path.split(".")[:-1])

                return
            else:
                logger.info("fetch \"%s\"", url)
                file_data = fetch_url(url)

                file_obj.file_path = os.path.join(file_obj.key_path, os.path.basename(url))

                with open(file_obj.file_path, "wb") as fh:  
                    fh.write(file_data)

                file_obj.file_url = url
                file_obj.version = version
                file_obj.ready = True

                return
        except Exception as e:
            logger.info("failed to fetch data file (\"%s\")", e)
            continue

    logger.error("failed to fetch data file \"%s\"", os.path.basename(file_obj.base_url))
