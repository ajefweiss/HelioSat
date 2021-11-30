# -*- coding: utf-8 -*-

"""datafile.py
"""

import cdflib
import datetime
import gzip
import heliosat
import logging
import numpy as np
import os
import shutil

from .transform import transform_reference_frame
from .util import dt_utc_from_str, fetch_url, url_regex_files, url_regex_resolve
from typing import List, Optional, Tuple
from netCDF4 import Dataset

class DataFile(object):
    """DataFile class.
    """
    base_urls: List[str]
    data_path: str
    data_key: str
    file_path: Optional[str]
    file_url: str
    key_path: str
    version: str

    ready: bool

    _json: dict

    def __init__(self, base_urls: List[str], filename: Optional[str], data_key: str, _json: dict) -> None:
        self.base_urls = base_urls
        self.data_path = os.getenv('HELIOSAT_DATAPATH', os.path.join(os.path.expanduser("~"), ".heliosat"))
        self.data_key = data_key
        self.file_path = None
        self.filename = filename
        self.key_path = os.path.join(self.data_path, "data", data_key)
        self._json = _json

        if not os.path.isdir(self.key_path):
            os.makedirs(self.key_path)

    def prepare(self, force_download: bool = False) -> None:
        logger = logging.getLogger(__name__)

        exception_list = []

        for base_url in self.base_urls:
            _version_list = list(self._json["keys"][self.data_key]["version_list"])
            _version_list.remove(self._json["keys"][self.data_key]["version_default"])

            version_list = [self._json["keys"][self.data_key]["version_default"]] + _version_list

            self.ready = False

            # check each version for local file
            for version in version_list:
                url = base_url.replace("{VER}", version)

                if self.filename:
                    filename = self.filename.replace("{VER}", version)
                else:
                    filename = None  # type: ignore

                try:
                    if url.startswith("$"):
                        # determine if any versions exist locally
                        if filename:
                            local_files = url_regex_files(filename, self.key_path)
                        else:
                            local_files = url_regex_files(url, self.key_path)

                        if len(local_files) > 0 and not force_download:
                            self.file_path = local_files[-1]
                            self.version = version
                            self.ready = True
                            return
                    else:
                        # determine if any versions exist locally
                        if filename:
                             self.file_path = os.path.join(self.key_path, filename)
                        else:
                            self.file_path = os.path.join(self.key_path, os.path.basename(url))

                        if os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0:  # type: ignore
                            self.version = version
                            self.ready = True

                            return
                except Exception as e:
                    exception_list.append(e)
                    continue

            # add functionality for remote compressed files
            if self._json["keys"][self.data_key].get("compression", None) == "gz":
                base_url = base_url + ".gz"

            # check each version for remote file
            _url_pre = None

            for version in version_list:
                    url = base_url.replace("{VER}", version)

                    # skip if url does not change with version
                    if url == _url_pre:
                        exception_list.append(version)
                        continue

                    _url_pre = url

                    if self.filename:
                        filename = self.filename.replace("{VER}", version)
                        self.file_path = os.path.join(self.key_path, filename)
                    else:
                        self.file_path = os.path.join(self.key_path, os.path.basename(url))  

                    try:
                        if url.startswith("$"):
                            url = str(url_regex_resolve(url, reduce=True))

                            file_data = fetch_url(url)

                            self.file_path = os.path.join(self.key_path, os.path.basename(url)) 

                            with open(self.file_path, "wb") as fh:
                                fh.write(file_data)

                            self.file_url = url
                            self.version = version
                            self.ready = True

                            # decompress
                            if self._json["keys"][self.data_key].get("compression", None) == "gz":
                                with gzip.open(self.file_path, "rb") as file_gz:
                                    with open(".".join(self.file_path.split(".")[:-1]), "wb") as file_extracted:
                                        shutil.copyfileobj(file_gz, file_extracted)

                                os.remove(self.file_path)

                                self.file_path = ".".join(self.file_path.split(".")[:-1])

                            return
                        else:
                            file_data = fetch_url(url)

                            self.file_path = os.path.join(self.key_path, os.path.basename(url))

                            with open(self.file_path, "wb") as fh:  # type: ignore
                                fh.write(file_data)

                            self.file_url = url
                            self.version = version
                            self.ready = True

                            return
                    except Exception as e:
                        exception_list.append(version)
                        continue
        
        logger.exception("failed to fetch data file \"%s\" (versions: %s)", os.path.basename(self.base_urls[0]), exception_list)
        self.ready = False
        

    def read(self, dt_start: datetime.datetime, dt_end: datetime.datetime, data_key: str, columns: List[str], reference_frame: str) -> Tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger(__name__)

        column_dicts = []

        # get version dict
        file_format = self._json["keys"][data_key]["format"]
        version_dict = self._json["keys"][data_key]["versions"].get(self.version, self._json["keys"][data_key]["versions"][self._json["keys"][data_key]["version_default"]])

        # resolve default columns
        if columns[0] == "~":
            default_columns = []

            for column in version_dict["data_columns"]:
                if column.get("load_by_default", False):
                    default_columns.append(column["names"][0])

            if len(columns) > 1:
                default_columns.extend(columns[1:])

            columns = default_columns

        # resolve alternative columns
        for i in range(len(columns)):
            valid_column = False

            for j in range(len(version_dict["data_columns"])):
                column = version_dict["data_columns"][j]

                if columns[i] in column.get("names", []):
                    columns[i] = column["names"][0]
                    column_dicts.append(column)
                    valid_column = True

                    break

            if not valid_column:
                raise KeyError("data column \"{0!s}\" is invalid ({1!s})".format(columns[i], self.file_path))

        if "_cdf" in file_format:
            dt_r, dk_r = self._read_cdf(dt_start, dt_end, version_dict, column_dicts, cdf_type=file_format)
        elif file_format == "tab":
            dt_r, dk_r = self._read_tab(dt_start, dt_end, version_dict, column_dicts)
        else:
            raise NotImplementedError("format \"{0!s}\" is not implemented".format(file_format))

        if dt_start == dt_end:
            dt_sel = np.argmin(np.abs(dt_r - dt_start.timestamp()))
            dt_mask = dt_r == dt_r[dt_sel]
        else:
            dt_mask = ((dt_r > dt_start.timestamp()) & (dt_r < dt_end.timestamp()))

        dt_r = dt_r[dt_mask]

        # process data columns
        for i in range(len(dk_r)):
            column = column_dicts[i]

            data_entry = dk_r[i][dt_mask]

            # filter values outside of range
            valid_range = column.get("valid_range", None)

            if valid_range:
                data_entry = np.where((data_entry > valid_range[0]) & (data_entry < valid_range[1]),
                                    data_entry, np.nan)

            # some data files aren't sorted by time
            sort_mask = np.argsort(dt_r)
            dt_r = dt_r[sort_mask]
            data_entry = data_entry[sort_mask]

            if data_entry.ndim == 1:
                data_entry = data_entry.reshape((-1, 1))

            # transform reference frame
            if reference_frame and "reference_frame" in column and reference_frame != column.get("reference_frame", None):
                heliosat._skm.reload()
                data_entry = transform_reference_frame(dt_r, data_entry, column["reference_frame"], reference_frame)

            dk_r[i] = data_entry

        return dt_r, np.concatenate(dk_r, axis=1)
        
    def _read_cdf(self, dt_start: datetime.datetime, dt_end: datetime.datetime, version_dict: dict, column_dicts: List[dict], cdf_type: str) -> Tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger(__name__)

        try:
            if cdf_type == "nasa_cdf":
                cdf_file = cdflib.CDF(self.file_path)
                epochs = cdf_file.varget(version_dict["time_column"]["key"])

                # special case when cdf files that have epoch = 0 entries
                if np.sum(epochs == 0) > 0:
                    null_filter = (epochs != 0)
                    epochs = epochs[null_filter]
                else:
                    null_filter = None

                dt_r = cdflib.epochs.CDFepoch.unixtime(epochs, to_np=True)
                dk_r = []

                for column in column_dicts:
                    key = column["key"]

                    if isinstance(key, str):
                        indices = column.get("indices", None)

                        if indices is not None:
                            indices = np.array(indices)
                            dk_r.append(np.array(cdf_file.varget(key)[:, indices]))
                        else:
                            dk_r.append(np.array(cdf_file.varget(key)))
                    elif isinstance(key, list):
                        dk_r.append(np.stack(arrays=[np.array(cdf_file.varget(k))
                                                     for k in key], axis=1))
                    else:
                        raise ValueError("cdf key must be a string or a list thereof")

                if null_filter is not None:
                    for i in range(len(dk_r)):
                        dk_r[i] = dk_r[i][null_filter]
            elif cdf_type == "net_cdf4":
                file = Dataset(self.file_path, "r")

                dt_r = np.array([t / 1000 for t in file.variables[version_dict["time_column"]["key"]][...]])
                dk_r = []

                for column in column_dicts:
                    key = column["key"]

                    if isinstance(key, str):
                        indices = column.get("indices", None)

                        if indices is not None:
                            indices = np.array(indices)
                            dk_r.append(np.array(file[key][:, indices]))
                        else:
                            dk_r.append(np.array(file[key][:]))
                    elif isinstance(key, list):
                        dk_r.append(np.stack(arrays=[np.array(file[k][:])
                                                    for k in key], axis=1))
                    else:
                        raise NotImplementedError
            else:
                raise ValueError("CDF type \"{0!r}\" is not supported".format(cdf_type))

            return dt_r, dk_r
        except Exception as e:
            logger.exception("failed to read file \"%s\" (%s)", self.file_path, e)
            raise Exception("failed to read file \"{0!s}\" ({1!r})".format(self.file_path, e))

    def _read_tab(self, dt_start: datetime.datetime, dt_end: datetime.datetime, version_dict: dict, column_dicts: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        logger = logging.getLogger(__name__)

        try:
            delimiter = version_dict["text_formatting"].get("delimiter", None)

            skip_cols = version_dict["text_formatting"].get("skip_columns", [])
            skip_rows = version_dict["text_formatting"].get("skip_rows", 0)

            time_format = version_dict["time_column"]["format"]
            time_index = version_dict["time_column"]["index"]

            converters = {}

            for col in skip_cols:
                converters[col] = lambda string: -1

            # ugly hack for badly formatted time strings
            def decode(string: str, format: str) -> float:
                if string.endswith("60.000"):
                    string = "{0}59.000".format(string[:-6])

                    return dt_utc_from_str(string, format).timestamp() + 1
                else:
                    return dt_utc_from_str(string, format).timestamp()

            if isinstance(time_format, str):
                converters[time_index] = lambda string: decode(string, time_format)   # type: ignore

                tab_file = np.loadtxt(self.file_path, skiprows=skip_rows, encoding="latin1", delimiter=delimiter,
                                      converters=converters)

                dt_r = tab_file[:, time_index]
            elif isinstance(time_format, int):
                tab_file = np.loadtxt(self.file_path, skiprows=skip_rows, converters=converters)

                dt_r = tab_file[:, time_index] + time_format
            else:
                logger.exception("time_format \"%s\" is not implemented", type(time_format))
                raise NotImplementedError("time_format \"{0!r}\" is not implemented".format(time_format))

            dk_r = []

            for column in column_dicts:
                indices = column["indices"]

                if isinstance(indices, int):
                    indices = column.get("indices", None)
                    indices = np.array(indices)
                    dk_r.append(np.stack([np.array(tab_file[:, index]) for index in indices]))

                elif isinstance(indices, list):
                    dk_r.append(np.stack([np.array(tab_file[:, index])
                                          for index in indices], axis=1))
                else:
                    raise NotImplementedError

            return dt_r, dk_r
        except Exception as e:
            logger.exception("failed to read file \"%s\" (%s)", self.file_path, e)
            raise Exception("failed to read file \"{0!s}\" ({1!r})".format(self.file_path, e))
