# -*- coding: utf-8 -*-

import datetime
import unittest

import heliosat


class TestSpacecraft(unittest.TestCase):
    def test_wind(self):
        wind_sp = heliosat.WIND()

        self.assertIsInstance(wind_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(wind_sp.spacecraft, dict)
        self.assertIsInstance(wind_sp.mission_start, datetime.datetime)
        self.assertIsInstance(wind_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2012, 1, 1)
        range_end = datetime.datetime(2012, 1, 3)

        time, data = wind_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = wind_sp.get_data(obs_dt, "mag", frame="HEEQ", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)

    def test_sta(self):
        sta_sp = heliosat.STA()

        self.assertIsInstance(sta_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(sta_sp.spacecraft, dict)
        self.assertIsInstance(sta_sp.mission_start, datetime.datetime)
        self.assertIsInstance(sta_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2012, 1, 1)
        range_end = datetime.datetime(2012, 1, 3)

        time, data = sta_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = sta_sp.get_data(obs_dt, "mag", frame="J2000", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)

    def test_stb(self):
        stb_sp = heliosat.STB()

        self.assertIsInstance(stb_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(stb_sp.spacecraft, dict)
        self.assertIsInstance(stb_sp.mission_start, datetime.datetime)
        self.assertIsInstance(stb_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2012, 1, 1)
        range_end = datetime.datetime(2012, 1, 3)

        time, data = stb_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = stb_sp.get_data(obs_dt, "mag", frame="J2000", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)

    def test_dscvor(self):
        dscovr_sp = heliosat.DSCOVR()

        self.assertIsInstance(dscovr_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(dscovr_sp.spacecraft, dict)
        self.assertIsInstance(dscovr_sp.mission_start, datetime.datetime)
        self.assertIsInstance(dscovr_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2019, 1, 1)
        range_end = datetime.datetime(2019, 1, 3)

        time, data = dscovr_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = dscovr_sp.get_data(obs_dt, "mag", frame="J2000", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)

    def test_vex(self):
        vex_sp = heliosat.VEX()

        self.assertIsInstance(vex_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(vex_sp.spacecraft, dict)
        self.assertIsInstance(vex_sp.mission_start, datetime.datetime)
        self.assertIsInstance(vex_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2012, 1, 1)
        range_end = datetime.datetime(2012, 1, 3)

        time, data = vex_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = vex_sp.get_data(obs_dt, "mag", frame="J2000", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)

    def test_mes(self):
        mes_sp = heliosat.MES()

        self.assertIsInstance(mes_sp, heliosat.spacecraft.Spacecraft)

        self.assertIsInstance(mes_sp.spacecraft, dict)
        self.assertIsInstance(mes_sp.mission_start, datetime.datetime)
        self.assertIsInstance(mes_sp.mission_end, datetime.datetime)

        range_start = datetime.datetime(2010, 1, 1)
        range_end = datetime.datetime(2010, 1, 3)

        time, data = mes_sp.get_data_raw(range_start, range_end, "mag")

        self.assertEqual(time.ndim, 1)
        self.assertEqual(data.ndim, 2)

        self.assertGreater(time.shape[0], 0)
        self.assertEqual(time.shape[0], data.shape[0])
        self.assertGreaterEqual(data.shape[1], 3)

        obs_dt = [range_start + datetime.timedelta(minutes=15 * i) for i in range(4 * 24 * 2)]

        obs_t, obs_d = mes_sp.get_data(obs_dt, "mag", frame="J2000", smoothing="kernel")

        self.assertGreater(obs_t[0], 0)
        self.assertEqual(obs_t.shape[0], len(obs_dt))
        self.assertEqual(obs_t.shape[0], obs_d.shape[0])
        self.assertGreaterEqual(obs_d.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
