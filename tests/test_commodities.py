"""Checks backtest_commodities' signals and trend accounting on made-up prices."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_commodities as bc  # noqa: E402
import backtest_orb as orb  # noqa: E402


def day(path):
    t = pd.Timestamp("2026-03-10 09:30")
    rows = list(path) + [path[-1][3:] * 4] * (78 - len(path))
    return orb.DayBars(pd.DataFrame([dict(hm=(t + pd.Timedelta(minutes=5 * i)).strftime("%H:%M"),
                                          open=o, high=h, low=l, close=c, volume=1e5, vwap=c)
                                     for i, (o, h, l, c) in enumerate(rows)]))


ORB = {"method": "orb", "stop_atr": 0.1, "be_r": None, "min_rvol": 1.5}
GAP = {"method": "gap", "gap_atr": 0.5, "stop_atr": 0.3}


class SignalTests(unittest.TestCase):
    def test_orb_long_breakout(self):
        b = day([(100, 100.5, 99.8, 100.4), (100.4, 100.45, 100.2, 100.3), (100.3, 101, 100.3, 100.9)])
        t = bc.day_signal(ORB, b, atr=2.0, prev_close=99.0, rvol=2.0)
        self.assertEqual((t["side"], t["k"], t["entry_raw"]), ("long", 2, 100.5))
        self.assertAlmostEqual(t["stop"], 100.3)

    def test_orb_needs_unusual_volume(self):
        b = day([(100, 100.5, 99.8, 100.4), (100.3, 101, 100.3, 100.9)])
        self.assertIsNone(bc.day_signal(ORB, b, atr=2.0, prev_close=99.0, rvol=1.2))

    def test_gap_up_is_faded_toward_yesterdays_close(self):
        b = day([(102, 102.2, 101.5, 101.6), (101.6, 101.7, 100.9, 101.0)])
        t = bc.day_signal(GAP, b, atr=2.0, prev_close=101.0, rvol=1.0)   # gap 1.0 = 0.5 ATR
        self.assertEqual((t["side"], t["entry_raw"], t["target"]), ("short", 102.0, 101.0))
        self.assertAlmostEqual(t["stop"], 102.6)
        xhm, xpx, why = bc.day_exit(b, 0, "short", t["stop"], t["target"], t["entry_raw"], None)
        self.assertEqual((xhm, why), ("09:35", "target"))

    def test_small_gap_is_ignored(self):
        b = day([(101.5, 101.6, 101.4, 101.5)])
        self.assertIsNone(bc.day_signal(GAP, b, atr=2.0, prev_close=101.0, rvol=1.0))


class TrendTests(unittest.TestCase):
    def test_channel_breakout_and_exit(self):
        cl = pd.Series([10, 10, 10, 11, 12, 11.5, 9], dtype=float)
        v = {"rule": "donch", "a": 3, "b": 2, "short": False}
        pos = bc.trend_targets(v, cl + 0.1, cl - 0.1, cl).tolist()
        self.assertEqual(pos, [0, 0, 0, 1, 1, 1, 0])

    def test_trend_profit_is_counted_from_entry_to_exit(self):
        dates = pd.date_range("2026-01-05", periods=8, freq="B")
        cl = pd.Series([10, 10, 10, 11, 12, 13, 9, 9], index=dates, dtype=float)
        frame = lambda s: pd.DataFrame({"X": s})  # noqa: E731
        ctx = {"daily": {"Open": frame(cl), "High": frame(cl + 0.1), "Low": frame(cl - 0.1), "Close": frame(cl)}}
        saved = bc.SLIP_PCT
        bc.SLIP_PCT = 0.0
        try:
            v = {"method": "trend", "rule": "donch", "a": 3, "b": 2, "short": False}
            trades, rows = bc.run_trend(v, "X", [d.strftime("%Y-%m-%d") for d in dates], ctx)
        finally:
            bc.SLIP_PCT = saved
        # breakout at the 11 close -> buy at the next open (12); exit signal at the 9 close -> sell at the next open (9)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["entry"], t["exit"]), (12.0, 9.0))
        self.assertAlmostEqual(rows[-1]["equity"] - bc.START_CASH, t["pnl"], places=2)


class CombineTests(unittest.TestCase):
    def test_daily_changes_add_up(self):
        a = [{"date": "d1", "equity": 101_000.0}, {"date": "d2", "equity": 101_000.0}]
        b = [{"date": "d1", "equity": 100_000.0}, {"date": "d2", "equity": 102_020.0}]
        rows = bc.combine([a, b], None, "")
        self.assertAlmostEqual(rows[0]["equity"], 101_000.0)
        self.assertAlmostEqual(rows[1]["equity"], 101_000.0 * 1.0202, places=1)


if __name__ == "__main__":
    unittest.main()
