HelioSat (ver 0.6)
========

A simple and small python package for handling and processing heliospheric satellite data. The current primary features are automatic data downloading & crude processing for Bepi, DSCOVR, PSP, SolO, STA, STB,and Wind. Furthermore all related and required SPICE kernels are downloaded automatically.

Installation
------------

Install the latest version manually using `git`:

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install .

or slightly older versions from PyPi with `pip install HelioSat`.

Basic Usage
-----------

Import the `heliosat` module and create a satellite instance:

    import heliosat

    wind_sat = heliosat.WIND()

This will automatically download and load any required SPICE kernels (using `spiceypy`). Note that
kernel or data files will be stored in `~/.heliosat` by default. As this may use up alot of disk
space you can alternatively change the default path by setting the environment variable `HELIOSAT_DATAPATH`.

Querying data (any tz-unaware datetime objects are assumed to be UTC) can then be done using:

    t_query = ["2020-01-01T00:00:00"]
    wind_t, wind_data = wind_sat.get(t_query, "mfi_h0", frame="GSE", return_datetimes=True)

By default only values at given datetimes are returned, if you need a time range you can either manually generate an extensive list of dates or use:

    t_query = ["2020-01-01T00:00:00", "2020-01-02T00:00:00"]
    wind_t, wind_data = wind_sat.get(t_query, "mfi_h0", as_endpoints=True, frame="GSE", return_datetimes=True)
