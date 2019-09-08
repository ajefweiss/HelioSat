HelioSat
========

A python package for handling and processing heliospheric satellite data.

Installation
------------

Install the latest version manually using `git` and `pip`:

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install .

Basic Usage
-----------

Simply import the module `heliosat` and create a satellite instance:

    import heliosat

    wind_sat = heliosat.WIND()

This will automatically download and load any required SPICE kernels (using `spiceypy`). Note that
kernel or data files will be stored in `~/.heliosat` by default. As this may use up alot of disk
space you can alternatively change the default path by setting the environment variable `HELIOSAT_DATAPATH`.

Querying the raw magnetic field data can then be done using:

    import datetime

    t_start = datetime.datetime(2010, 1, 1)
    t_end = datetime.datetime(2010, 1, 3)

    t_raw, data_raw = wind_sat.get_data_raw(t_start, t_end, "mag")

or alternatively processed data at specific times in a specific reference frame:

    # observer datetimes for an entire week
    obs = [t_start + datetime.timedelta(minutes=12 * i) for i in range(0, 7 * 24 * 5)]

    # smoothing using a gaussian kernel and a smoothing scale of 5 minutes, the data is also cached
    t_sm, data_sm = heliosat.get_data(obs, "mag", frame="J2000", smoothing="kernel", cache=True)

Features
--------

SPICE:

 - Get object trajectory using `wind_sat.trajectory(t, frame="J2000")`

Available data: 

 - Magnetic field data in chosen reference frame (`data="mag"`)
 - Proton data as density, speed & temperature (`data="proton"`)

Available satellites & data:

| Satellites | Trajectory | Magnetic Field | Protons |
| ---------- |:----------:|:--------------:|:-------:|
| DSCOVR     | No*        | Yes            | Yes     |
| MESSENGER  | Yes        | Yes            | No      |
| PSP        | Yes        | No             | No      |
| STEREO A   | Yes        | Yes            | Yes     |
| STEREO B   | Yes        | Yes            | Yes     |
| VEX        | Yes        | Yes            | No      |
| WIND       | No*        | Yes            | Yes     |

**DSCOVR and WIND trajectory currently return the Earth trajectory.*