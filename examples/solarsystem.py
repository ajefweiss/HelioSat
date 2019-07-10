# -*- coding: utf-8 -*-

import datetime
import heliosat

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect("equal")
    ax.set_title("Solar System (last 200 days)")

    times = [datetime.datetime(2018, 1, 1) - datetime.timedelta(days=5 * k) for k in range(40)]

    sun = heliosat.Sun.trajectory(times)
    mercury = heliosat.Mercury.trajectory(times)
    venus = heliosat.Venus.trajectory(times)
    earth = heliosat.Earth.trajectory(times)
    mars = heliosat.Mars.trajectory(times)
    jupiter = heliosat.Jupiter.trajectory(times)

    ax.scatter(sun[0, 0], sun[0, 1], sun[0, 1], color="y", alpha=1)
    ax.scatter(mercury[0, 0], mercury[0, 1], mercury[0, 1], color="b", alpha=1)
    ax.scatter(venus[0, 0], venus[0, 1], venus[0, 1], color="m", alpha=1)
    ax.scatter(earth[0, 0], earth[0, 1], earth[0, 1], color="g", alpha=1)
    ax.scatter(mars[0, 0], mars[0, 1], mars[0, 1], color="r", alpha=1)
    ax.scatter(jupiter[0, 0], jupiter[0, 1], jupiter[0, 1], color="k", alpha=1)

    ax.plot(mercury[:, 0], mercury[:, 1], mercury[:, 1], "b-", alpha=0.8)
    ax.plot(venus[:, 0], venus[:, 1], venus[:, 1], "m-", alpha=0.8)
    ax.plot(earth[:, 0], earth[:, 1], earth[:, 1], "g-", alpha=0.8)
    ax.plot(mars[:, 0], mars[:, 1], mars[:, 1], "r-", alpha=0.8)
    ax.plot(jupiter[:, 0], jupiter[:, 1], jupiter[:, 1], "k-", alpha=0.8)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()
