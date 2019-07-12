# -*- coding: utf-8 -*-

import datetime
import heliosat
import numpy as np

from matplotlib import pyplot as plt


if __name__ == "__main__":
    heliosat.configure_logging()

    dscovr = heliosat.DSCOVR()

    start_time = datetime.datetime(2018, 5, 16) - datetime.timedelta(hours=6)
    stop_time = datetime.datetime(2018, 5, 19) + datetime.timedelta(hours=6)

    smooth_times = [start_time + datetime.timedelta(hours=6, minutes=i) for i in range(3 * 24 * 60)]

    dscovr_raw_times, dscovr_raw_fc_data = dscovr.get_fc(start_time, stop_time)
    dscovr_times = [datetime.datetime.fromtimestamp(t) for t in dscovr_raw_times]

    fig, axes = plt.subplots(3, 6, figsize=(12, 8), sharex=True, sharey=False,
                             gridspec_kw={"wspace": 0, "hspace": 0})

    print(np.min(dscovr_raw_fc_data))

    scales = [600, 1800, 3600]

    axes[0][0].set_title("Average (n)")
    axes[0][1].set_title("Adaptive (n)")

    axes[0][2].set_title("Average (V)")
    axes[0][3].set_title("Adaptive (V)")

    axes[0][4].set_title("Average (T)")
    axes[0][5].set_title("Adaptive (T)")

    for i in range(0, len(scales)):
        axes[i][2].set_ylabel("Scale: {0:.0f}s".format(scales[i]))
        axes[i][2].yaxis.set_label_position("right")

    # average smoothing
    for i in range(0, 3):
        smooth_fc_data = dscovr.get_fc_proc(smooth_times, smoothing="average",
                                            smoothing_scale=scales[i],
                                            raw_data=(dscovr_raw_times, dscovr_raw_fc_data))

        axes[i][0].plot(dscovr_times, dscovr_raw_fc_data[:, 0], "b-", alpha=0.4)
        axes[i][2].plot(dscovr_times, dscovr_raw_fc_data[:, 1], "g-", alpha=0.4)
        axes[i][4].plot(dscovr_times, dscovr_raw_fc_data[:, 2], "r-", alpha=0.4)

        axes[i][0].plot(smooth_times, smooth_fc_data[:, 0], "b--", alpha=0.8)
        axes[i][2].plot(smooth_times, smooth_fc_data[:, 1], "g--", alpha=0.8)
        axes[i][4].plot(smooth_times, smooth_fc_data[:, 2], "r--", alpha=0.8)

        axes[i][0].grid(True)
        axes[i][2].grid(True)
        axes[i][4].grid(True)

    # adaptive gaussian smoothing
    for i in range(0, 3):
        smooth_fc_data = dscovr.get_fc_proc(smooth_times, smoothing="adaptive_gaussian",
                                            smoothing_scale=scales[i],
                                            raw_data=(dscovr_raw_times, dscovr_raw_fc_data))

        axes[i][1].plot(dscovr_times, dscovr_raw_fc_data[:, 0], "b-", alpha=0.4)
        axes[i][3].plot(dscovr_times, dscovr_raw_fc_data[:, 1], "g-", alpha=0.4)
        axes[i][5].plot(dscovr_times, dscovr_raw_fc_data[:, 2], "r-", alpha=0.4)

        axes[i][1].plot(smooth_times, smooth_fc_data[:, 0], "b--", alpha=0.8)
        axes[i][3].plot(smooth_times, smooth_fc_data[:, 1], "g--", alpha=0.8)
        axes[i][5].plot(smooth_times, smooth_fc_data[:, 2], "r--", alpha=0.8)

        axes[i][1].grid(True)
        axes[i][3].grid(True)
        axes[i][5].grid(True)

    # plot all figures
    fig.autofmt_xdate()
    plt.show()
