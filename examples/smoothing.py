# -*- coding: utf-8 -*-

import datetime
import heliosat
import numpy as np

from matplotlib import pyplot as plt


if __name__ == "__main__":
    heliosat.configure_logging()

    wind = heliosat.DSCOVR()

    start_time = datetime.datetime(2017, 5, 16) - datetime.timedelta(hours=6)
    stop_time = datetime.datetime(2017, 5, 19) + datetime.timedelta(hours=6)

    smooth_times = [start_time + datetime.timedelta(hours=6, minutes=i) for i in range(3 * 24 * 60)]

    wind_raw_times, wind_raw_mag_data = wind.get_mag(start_time, stop_time, stride=8)
    wind_times = [datetime.datetime.fromtimestamp(t) for t in wind_raw_times]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})

    scales = [600, 1800, 3600]

    axes[0][0].set_title("Moving Average")
    axes[0][1].set_title("Adaptive Gaussian")
    axes[0][2].set_title("Adaptive Gaussian Normalized")

    for i in range(0, len(scales)):
        axes[i][0].set_ylabel("nT")
        axes[i][2].set_ylabel("Scale: {0:.0f}s".format(scales[i]))
        axes[i][2].yaxis.set_label_position("right")

    # average smoothing
    for i in range(0, 3):
        smooth_mag_data = wind.get_mag_proc(smooth_times, smoothing="average",
                                            smoothing_scale=scales[i],
                                            raw_data=(wind_raw_times, wind_raw_mag_data))

        axes[i][0].plot(wind_times, np.sqrt(np.sum(wind_raw_mag_data ** 2, axis=1)), "k-",
                        alpha=0.3)
        axes[i][0].plot(wind_times, wind_raw_mag_data[:, 0], "b-", alpha=0.4)
        axes[i][0].plot(wind_times, wind_raw_mag_data[:, 1], "g-", alpha=0.4)
        axes[i][0].plot(wind_times, wind_raw_mag_data[:, 2], "r-", alpha=0.4)

        axes[i][0].plot(smooth_times, np.sqrt(np.sum(smooth_mag_data ** 2, axis=1)), "k-",
                        alpha=0.7)
        axes[i][0].plot(smooth_times, smooth_mag_data[:, 0], "b--", alpha=0.8)
        axes[i][0].plot(smooth_times, smooth_mag_data[:, 1], "g--", alpha=0.8)
        axes[i][0].plot(smooth_times, smooth_mag_data[:, 2], "r--", alpha=0.8)

        axes[i][0].grid(True)

    # adaptive gaussian smoothing
    for i in range(0, 3):
        smooth_mag_data = wind.get_mag_proc(smooth_times, smoothing="adaptive_gaussian",
                                            smoothing_scale=scales[i],
                                            raw_data=(wind_raw_times, wind_raw_mag_data))

        axes[i][1].plot(wind_times, np.sqrt(np.sum(wind_raw_mag_data ** 2, axis=1)), "k-",
                        alpha=0.3)
        axes[i][1].plot(wind_times, wind_raw_mag_data[:, 0], "b-", alpha=0.4)
        axes[i][1].plot(wind_times, wind_raw_mag_data[:, 1], "g-", alpha=0.4)
        axes[i][1].plot(wind_times, wind_raw_mag_data[:, 2], "r-", alpha=0.4)

        axes[i][1].plot(smooth_times, np.sqrt(np.sum(smooth_mag_data ** 2, axis=1)), "k-",
                        alpha=0.7)
        axes[i][1].plot(smooth_times, smooth_mag_data[:, 0], "b--", alpha=0.8)
        axes[i][1].plot(smooth_times, smooth_mag_data[:, 1], "g--", alpha=0.8)
        axes[i][1].plot(smooth_times, smooth_mag_data[:, 2], "r--", alpha=0.8)

        axes[i][1].grid(True)

    # adaptive gaussian normalized smoothing
    for i in range(0, 3):
        smooth_mag_data = wind.get_mag_proc(smooth_times, smoothing="adaptive_gaussian_normalized",
                                            smoothing_scale=scales[i],
                                            raw_data=(wind_raw_times, wind_raw_mag_data))

        axes[i][2].plot(wind_times, np.sqrt(np.sum(wind_raw_mag_data ** 2, axis=1)), "k-",
                        alpha=0.3)
        axes[i][2].plot(wind_times, wind_raw_mag_data[:, 0], "b-", alpha=0.4)
        axes[i][2].plot(wind_times, wind_raw_mag_data[:, 1], "g-", alpha=0.4)
        axes[i][2].plot(wind_times, wind_raw_mag_data[:, 2], "r-", alpha=0.4)

        axes[i][2].plot(smooth_times, np.sqrt(np.sum(smooth_mag_data ** 2, axis=1)), "k-",
                        alpha=0.7)
        axes[i][2].plot(smooth_times, smooth_mag_data[:, 0], "b--", alpha=0.8)
        axes[i][2].plot(smooth_times, smooth_mag_data[:, 1], "g--", alpha=0.8)
        axes[i][2].plot(smooth_times, smooth_mag_data[:, 2], "r--", alpha=0.8)

        axes[i][2].grid(True)

    # plot all figures
    fig.autofmt_xdate()
    plt.show()
