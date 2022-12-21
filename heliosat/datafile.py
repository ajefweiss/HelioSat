# -*- coding: utf-8 -*-

"""datafile.py
"""

import datetime as dt
import gzip
import logging as lg
import os
import re
import shutil
from typing import List, Optional, Tuple

import cdflib
import numpy as np
from netCDF4 import Dataset

import heliosat

from .transform import transform_reference_frame
from .util import (dt_utc_from_str, fetch_url, url_regex_files,
                   url_regex_resolve)


class DataFile(object):
    """DataFile class."""

    base_url: str
    data_path: str
    data_key: str
    file_path: Optional[str]
    file_url: str
    key_path: str
    version: str

    ready: bool

    _json: dict

    def __init__(self, base_url: str, file_expr: str, data_key: str, _json: dict) -> None:
        self.base_url = base_url
        self.file_expr = file_expr
        self.data_path = os.getenv("HELIOSAT_DATAPATH", os.path.join(os.path.expanduser("~"), ".heliosat"))
        self.data_key = data_key
        self.file_path = None
        self.key_path = os.path.join(self.data_path, "data", data_key)
        self._json = _json

        if not os.path.isdir(self.key_path):
            os.makedirs(self.key_path)

    def download(self) -> Tuple[np.ndarray, np.ndarray]:
        logger = lg.getLogger(__name__)

        url = self.base_url

        try:
            if url.startswith("$"):
                results, groups = url_regex_resolve(url)

                # groups should be versions, at minimum numbers
                versions = [int(_) for _ in groups]

                result_newest = [res for _, res in sorted(zip(versions, results))][-1]
                version_newest = sorted(versions)[-1]

                logger.info('fetch "%s"', result_newest)
                file_data = fetch_url(result_newest)

                file_path = os.path.join(self.key_path, os.path.basename(result_newest))

                with open(file_path, "wb") as fh:
                    fh.write(file_data)

                # decompress
                if self._json["keys"][self.data_key].get("compression", None) == "gz":
                    with gzip.open(file_path, "rb") as file_gz:
                        with open(".".join(file_path.split(".")[:-1]), "wb") as file_extracted:
                            shutil.copyfileobj(file_gz, file_extracted)

                    os.remove(file_path)
                    file_path = ".".join(file_path.split(".")[:-1])

                self.file_path = file_path
                self.file_url = result_newest
                self.version = version_newest
                self.ready = True

                return True
            else:
                logger.info('fetch w/ headers "%s"', url)
                file_data, headers = fetch_url(url, return_headers=True)

                if self.file_expr:
                    match = re.search(self.file_expr, headers["Content-Disposition"])

                    file_path = os.path.join(self.key_path, match.group())

                    version = match.groups()[0]
                else:
                    file_path = os.path.join(self.key_path, os.path.basename(url))

                    version = None

                with open(file_path, "wb") as fh:
                    fh.write(file_data)

                self.file_path = file_path
                self.file_url = url
                self.version = version
                self.ready = True

                return True
        except Exception as e:
            logger.info('failed to fetch data file ("%s")', e)
            return False

    def local(self) -> Optional[str]:
        logger = lg.getLogger("__name__")

        try:
            if self.base_url.startswith("$"):
                local_files, local_version = url_regex_files(self.base_url, self.key_path)

                if len(local_files) > 0:
                    self.file_path = local_files[-1]
                    self.version = local_version[-1]
                    self.ready = True

                    return True
            else:
                if self.file_expr:
                    local_files, local_version = url_regex_files(self.file_expr, self.key_path)
                    if len(local_files) > 0:
                        self.file_path = local_files[-1]
                        self.version = local_version[-1]
                        self.ready = True

                        return True
                else:
                    local_file = os.path.join(self.key_path, os.path.basename(self.base_url))

                    if os.path.isfile(local_file) and os.path.getsize(local_file) > 0:
                        self.file_path = local_file
                        self.version = None
                        self.ready = True

                        return True
        except Exception as e:
            logger.warning("exception when checking local file %s (%s)", self.base_url, e)
            return False

        return False

    def read(
        self,
        dt_start: dt.datetime,
        dt_end: dt.datetime,
        data_key: str,
        columns: List[str],
        kernel_group: str,
        reference_frame: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        column_dicts = []

        file_format = self._json["keys"][data_key]["format"]

        # determine version
        all_keys = list(self._json["keys"][data_key]["versions"].keys())

        if self.version:
            # find highest version that is equal or below file version, must be integers
            all_keys_inv = all_keys[::-1]

            i = 0

            while int(all_keys_inv[i]) > int(self.version):
                i += 1

            version_dict = self._json["keys"][data_key]["versions"][all_keys_inv[i]]

        else:
            # use lastest version
            version_dict = self._json["keys"][data_key]["versions"][all_keys[-1]]

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
                raise KeyError('data column "{0!s}" is invalid ({1!s})'.format(columns[i], self.file_path))

        if "_cdf" in file_format:
            dtp_r, dk_r = self._read_cdf(dt_start, dt_end, version_dict, column_dicts, cdf_type=file_format)
        elif file_format == "tab":
            dtp_r, dk_r = self._read_tab(dt_start, dt_end, version_dict, column_dicts)
        else:
            raise NotImplementedError('format "{0!s}" is not implemented'.format(file_format))

        if dt_start == dt_end:
            dt_sel = np.argmin(np.abs(dtp_r - dt_start.timestamp()))
            dt_mask = dtp_r == dtp_r[dt_sel]
        else:
            dt_mask = (dtp_r > dt_start.timestamp()) & (dtp_r < dt_end.timestamp())

        dtp_r = dtp_r[dt_mask]

        # process data columns
        for i in range(len(dk_r)):
            column = column_dicts[i]

            data_entry = dk_r[i][dt_mask]

            # filter values outside of range
            valid_range = column.get("valid_range", None)

            if valid_range:
                data_entry = np.where(
                    (data_entry > valid_range[0]) & (data_entry < valid_range[1]),
                    data_entry,
                    np.nan,
                )

            # some data files aren't sorted by time
            sort_mask = np.argsort(dtp_r)
            dtp_r = dtp_r[sort_mask]
            data_entry = data_entry[sort_mask]

            if data_entry.ndim == 1:
                data_entry = data_entry.reshape((-1, 1))

            # transform reference frame
            if len(dtp_r) > 0 and reference_frame and reference_frame != column.get("reference_frame", None):
                heliosat._skm.load_group("default")

                if kernel_group:
                    heliosat._skm.load_group(kernel_group)

                data_entry = transform_reference_frame(dtp_r, data_entry, column["reference_frame"], reference_frame)

            dk_r[i] = data_entry

        return dtp_r, np.concatenate(dk_r, axis=1)

    def _read_cdf(
        self, dt_start: dt.datetime, dt_end: dt.datetime, version_dict: dict, column_dicts: List[dict], cdf_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if cdf_type == "nasa_cdf":
                cdf_file = cdflib.CDF(self.file_path)
                epochs = cdf_file.varget(version_dict["time_column"]["key"])

                # special case when cdf files that have epoch = 0 entries
                if np.sum(epochs == 0) > 0:
                    null_filter = epochs != 0
                    epochs = epochs[null_filter]
                else:
                    null_filter = None

                dtp_r = cdflib.epochs.CDFepoch.unixtime(epochs, to_np=True)
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
                        dk_r.append(
                            np.stack(
                                arrays=[np.array(cdf_file.varget(k)) for k in key],
                                axis=1,
                            )
                        )
                    else:
                        raise ValueError("cdf key must be a string or a list thereof")

                if null_filter is not None:
                    for i in range(len(dk_r)):
                        dk_r[i] = dk_r[i][null_filter]
            elif cdf_type == "net_cdf4":
                file = Dataset(self.file_path, "r")

                dtp_r = np.array([t / 1000 for t in file.variables[version_dict["time_column"]["key"]][...]])
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
                        dk_r.append(np.stack(arrays=[np.array(file[k][:]) for k in key], axis=1))
                    else:
                        raise NotImplementedError
            else:
                raise ValueError('CDF type "{0!r}" is not supported'.format(cdf_type))

            return dtp_r, dk_r
        except Exception as e:
            raise Exception('failed to read file "{0!s}" ({1!r})'.format(self.file_path, e))

    def _read_tab(
        self, dt_start: dt.datetime, dt_end: dt.datetime, version_dict: dict, column_dicts: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                converters[time_index] = lambda string: decode(string, time_format)

                tab_file = np.loadtxt(
                    self.file_path,
                    skiprows=skip_rows,
                    encoding="latin1",
                    delimiter=delimiter,
                    converters=converters,
                )

                dtp_r = tab_file[:, time_index]
            elif isinstance(time_format, int):
                tab_file = np.loadtxt(self.file_path, skiprows=skip_rows, converters=converters)

                dtp_r = tab_file[:, time_index] + time_format
            else:
                raise NotImplementedError('time_format "{0!r}" is not implemented'.format(time_format))

            dk_r = []

            for column in column_dicts:
                indices = column["indices"]

                if isinstance(indices, int):
                    indices = column.get("indices", None)
                    indices = np.array(indices)
                    dk_r.append(np.stack([np.array(tab_file[:, index]) for index in indices]))

                elif isinstance(indices, list):
                    dk_r.append(np.stack([np.array(tab_file[:, index]) for index in indices], axis=1))
                else:
                    raise NotImplementedError

            return dtp_r, dk_r
        except Exception as e:
            raise Exception('failed to read file "{0!s}" ({1!r})'.format(self.file_path, e))
