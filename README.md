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

This will automatically download and load any required SPICE kernels (using `spiceypy`).

One can then query raw magnetic field data using:

    import datetime

    t_start = datetime.datetime(2010, 1, 1)
    t_end = datetime.datetime(2010, 1, 3)

    raw_data = wind_sat.get_data_raw(t_start, t_end, data="mag")

or alternatively smoothed data:

    t = [t_start + datetime.timedelta(minutes=12 * i) for i in range(0, 360)]

    # smoothing using a gaussian kernel and a scale of 20 minutes
    smooth_data = heliosat.smooth_gaussian_1d(t, raw_data[0], raw_data[1], smoothing_scale=1200)

Features
--------

SPICE:

 - Get object trajectory using `wind_sat.trajectory(t, reference_frame="GSE")`

Available data: 

 - Magnetic field data in chosen reference frame (`data="mag"`)
 - Proton data as density, speed & temperature (`data="proton"`)

Available satellites & data:

| Satellites | Trajectory | Magnetic Field | Protons |
| ---------- |:----------:|:--------------:|:-------:|
| DSCOVR     | No*        | Yes            | Yes     |
| MESSENGER  | Yes        | Yes            | No      |
| PSP        | Yes        | Yes            | No      |
| STEREO A   | Yes        | Yes            | Yes     |
| STEREO B   | Yes        | Yes            | Yes     |
| VEX        | Yes        | Yes            | No      |
| WIND       | No*        | Yes            | No      |

**DSCOVR and WIND trajectory currently return the Earth trajectory.*