HelioSat
========

A simple and small python package for handling and processing heliospheric satellite data. The current primary features are automatic data downloading & crude processing for DSCOVR, MES, PSP, STA, STB, VEX and WIND (plus BEPI and SOLO once data products are publicly available). Furthermore all related and required SPICE kernels are downloaded automatically.

Installation
------------

Install the latest version manually using `git`:

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install .

or from PyPi with `pip install HelioSat`.

Basic Usage
-----------

Import the `heliosat` module and create a satellite instance:

    import heliosat

    wind_sat = heliosat.WIND()

This will automatically download and load any required SPICE kernels (using `spiceypy`). Note that
kernel or data files will be stored in `~/.heliosat` by default. As this may use up alot of disk
space you can alternatively change the default path by setting the environment variable `HELIOSAT_DATAPATH`.

Querying raw data in a certain time window (any tz-unaware datetime objects are assumed to be UTC) can then be done using:

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