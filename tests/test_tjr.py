"""Checks backtest_tjr's sweep -> structure shift -> fair value gap rules on a made-up morning."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_tjr as tjr  # noqa: E402


def day(rows, start="09:00"):
    t = pd.Timestamp("2026-03-10 " + start)
    return pd.DataFrame([dict(hm=(t + pd.Timedelta(minutes=5 * i)).strftime("%H:%M"), open=o, high=h, low=l, close=c)
                         for i, (o, h, l, c) in enumerate(rows)])


# 9:00-9:25 pre-market drifts 100.2-100.6 with a swing high of 100.6 at 9:10; pre-market low 100.0.
PRE = [(100.3, 100.4, 100.2, 100.3), (100.3, 100.5, 100.25, 100.45), (100.45, 100.6, 100.4, 100.5),
       (100.5, 100.52, 100.3, 100.35), (100.35, 100.4, 100.0, 100.1), (100.1, 100.3, 100.05, 100.2)]
# 9:30 sweep: wick to 99.8 below the 100.0 low, close back above at 100.15.
# 9:35-9:45 displacement up with a gap (9:35 high 100.3 < 9:45 low 100.5) and a close above 100.6 (MSS at 9:45).
# 9:50 pulls back into the gap (low 100.45 <= 100.5) -> long fills at 100.5. Stop 99.8 x (1 - 0.0002) ~ 99.78.
# Target = 100.5 + 2 x 0.72 ~ 101.94, reached at 10:30.
MORNING = [(100.2, 100.25, 99.8, 100.15), (100.15, 100.3, 100.1, 100.28), (100.28, 100.55, 100.27, 100.5),
           (100.55, 100.8, 100.5, 100.75), (100.75, 100.76, 100.45, 100.6)]
RALLY = [(100.6 + 0.2 * i, 100.8 + 0.2 * i, 100.55 + 0.2 * i, 100.75 + 0.2 * i) for i in range(10)]
REST = [(102.5, 102.6, 102.4, 102.5)] * 60


class TjrRuleTests(unittest.TestCase):
    def test_textbook_long(self):
        b = day(PRE + MORNING + RALLY + REST)
        pools = {"low": [100.0], "high": [103.0]}
        setups = tjr.find_setups(b, pools)
        self.assertEqual(len(setups), 1)
        s = setups[0]
        self.assertEqual((s["side"], s["sweep_hm"], s["mss_hm"]), ("long", "09:30", "09:45"))
        self.assertAlmostEqual(s["entry"], 100.5)
        self.assertAlmostEqual(s["stop"], 99.8 * (1 - 0.0002))
        r = tjr.fill_and_exit(b, s)
        self.assertEqual(r[0], "09:50")             # filled on the pullback into the gap
        self.assertAlmostEqual(r[1], 100.5)
        self.assertEqual(r[4], "target")

    def test_breakout_that_holds_is_not_a_sweep(self):
        # 9:30 closes BELOW the pre-market low: a real breakdown, not a sweep
        bad = [(100.2, 100.25, 99.6, 99.7)] + MORNING[1:]
        b = day(PRE + bad + RALLY + REST)
        self.assertEqual(tjr.find_setups(b, {"low": [100.0], "high": [103.0]}), [])

    def test_no_setup_outside_the_killzone(self):
        flat = [(100.3, 100.35, 100.25, 100.3)] * 24          # 9:30-11:25 quiet
        late = [(100.2, 100.25, 99.8, 100.15)] + MORNING[1:]   # same pattern at 11:30
        b = day(PRE + flat + late + REST)
        self.assertEqual(tjr.find_setups(b, {"low": [100.0], "high": [103.0]}), [])


if __name__ == "__main__":
    unittest.main()
