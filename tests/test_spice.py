# -*- coding: utf-8 -*-

import datetime
import numpy as np
import os
import spiceypy
import unittest

import heliosat


class TestSpice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        heliosat.spice.spice_init()

    def test_planets(self):
        self.assertIsInstance(heliosat.Sun(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Mercury(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Venus(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Earth(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Moon(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Mars(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Jupiter(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Saturn(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Uranus(), heliosat._SpiceObject)
        self.assertIsInstance(heliosat.Neptune(), heliosat._SpiceObject)

    def test_kernel_dict(self):
        self.assertIsInstance(heliosat._spice, dict)

        self.assertGreater(len(heliosat._spice["kernel_groups"]), 0)
        self.assertGreater(len(heliosat._spice["kernels_loaded"]), 0)

    def test_earth_trajectory_24h(self):
        times = [datetime.datetime(2010, 1, 1, i) for i in range(0, 24)]
        earth_trajectory = heliosat.Earth().trajectory(times)

        self.assertIsInstance(earth_trajectory, np.ndarray)
        self.assertEqual(earth_trajectory.shape, (24, 3))

    def test_transform_frame(self):
        data = np.random.rand(10, 3)
        ts = [datetime.datetime(2000, 6, 5 + i) for i in range(10)]

        self.assertEqual(heliosat.spice.transform_frame(ts, data, "J2000", "HEEQ").shape, (10, 3))
        self.assertEqual(
            heliosat.spice.transform_frame_lonlat(ts, data[:, :2], "J2000", "HEEQ").shape, (10, 2)
        )

    @classmethod
    def tearDownClass(cls):
        kernels_path = os.path.join(os.getenv('HELIOSAT_DATAPATH',
                                    os.path.join(os.path.expanduser("~"), ".heliosat")), "kernels")

        for kernel_url in heliosat._spice["kernels_loaded"]:
            spiceypy.unload(os.path.join(kernels_path, kernel_url.split("/")[-1]))

        heliosat._spice = None


if __name__ == "__main__":
    unittest.main()
