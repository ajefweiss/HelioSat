HelioSat
========

A small simple python package for handling and processing heliospheric satellite data.

Installation
------------

Install the latest version manually using `git`:

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install .

or from PyPi.

Basic Usage
-----------

Simply import the module `heliosat` and create a satellite instance:

    import heliosat

    wind_sat = heliosat.WIND()

This will automatically download and load any required SPICE kernels (using `spiceypy`). Note that
kernel or data files will be stored in `~/.heliosat` by default. As this may use up alot of disk
space you can alternatively change the default path by setting the environment variable `HELIOSAT_DATAPATH`.

Querying raw data in a certain time window can then be done using:

    import datetime

    t_start = datetime.datetime(2010, 1, 1)
    t_end = datetime.datetime(2010, 1, 3)

    t_raw, data_raw = wind_sat.get_data_raw(t_start, t_end, "mfi_h0")

Alternatively processed data at specific times in a specific reference frame can be queried using:

    # observer datetimes for an entire week
    obs = [t_start + datetime.timedelta(minutes=12 * i) for i in range(0, 7 * 24 * 5)]

    # smoothing using a gaussian kernel and a smoothing scale of 5 minutes, the data is also cached
    t_sm, data_sm = heliosat.get_data(obs, "mfi_h0", frame="J2000", smoothing="kernel", cache=True,
                                      return_datetimes=True, remove_nans=True)

If particular data columns are not being read out but are present within the data files, they can be added by setting the `extra_columns` parameter.

Features
--------

By default most satellites will have "mag" and "proton" data available. For a full list of all available data and definitions see the heliosat/json/spacecraft.json file.

SPICE:

 - Get object trajectory using `wind_sat.trajectory(obs, frame="J2000")`

Available satellites & data:

| Satellites | Trajectory | Magnetic Field | Protons |
| ---------- |:----------:|:--------------:|:-------:|
| DSCOVR     | No*        | Yes            | Yes     |
| MESSENGER  | Yes        | Yes            | No      |
| PSP        | Yes        | Yes            | Yes     |
| STEREO A   | Yes        | Yes            | Yes     |
| STEREO B   | Yes        | Yes            | Yes     |
| VEX        | Yes        | Yes            | No      |
| WIND       | No*        | Yes            | Yes     |

**DSCOVR and WIND trajectory currently return the Earth trajectory.*