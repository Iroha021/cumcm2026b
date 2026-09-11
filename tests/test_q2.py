"""问题二自检：正交律、禁飞走廊、候选区域、批量几何与逐条裁剪一致。"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import geom, policy as policy_mod  # noqa: E402
from run import load_config  # noqa: E402

CFG = load_config()
DELTA = float(CFG["env"]["bearing_error_deg"])


class TestBatchGeometry(unittest.TestCase):
    def test_batch_matches_exact_region(self):
        """批量直径（向量化）必须与 geom 的逐条裁剪结果一致。"""
        P1, th1 = (500.0, 300.0), 30.0
        P2 = (-300.0, 900.0)
        th2 = np.linspace(0.0, 359.0, 90)
        D = policy_mod.batch_region_diameter(P1, th1, P2, th2, DELTA)
        bad = 0
        for i, t in enumerate(th2):
            reg = geom.region_from_bearings([(P1, th1), (P2, float(t))], DELTA)
            if reg.valid:
                if abs(D[i] - reg.diameter()) > 1e-6:
                    bad += 1
            else:
                if D[i] < policy_mod.PENALTY_D - 1e-9:
                    bad += 1
        self.assertEqual(bad, 0)


class TestSecondPoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = policy_mod.select_second_point(
            CFG, (500.0, 300.0), 30.0,
            rng=np.random.default_rng(7), n_mc=400,
            beta_step_deg=10.0, d_step_m=200.0)

    def test_returns_valid_plan(self):
        p = self.plan
        self.assertIn("P2", p)
        self.assertGreater(p["pi"], 0.5, "第二点必须大概率能测到")
        self.assertGreater(p["E_D"], 0.0)
        self.assertLess(p["E_D"], 400.0)

    def test_avoids_collinear_corridor(self):
        """不能选在第一示向线上（β≈0 会让定位区域发散）。"""
        self.assertGreater(self.plan["beta"], 10.0)
        self.assertGreater(self.plan["phi"], float(CFG["q2"]["phi_min_deg"]))

    def test_candidate_region_contains_optimum(self):
        cr = self.plan["candidate_region"]
        self.assertIsNotNone(cr["beta_min"])
        self.assertLessEqual(cr["beta_min"], self.plan["beta"] + 1e-9)
        self.assertGreaterEqual(cr["beta_max"], self.plan["beta"] - 1e-9)
        self.assertLessEqual(cr["d_min"], self.plan["d"])
        self.assertGreaterEqual(cr["d_max"], self.plan["d"])

    def test_distance_scales_with_prior(self):
        """最优距离应与源距离后验均值同量级（约 1~1.5 倍）。"""
        r_hat = self.plan["prior_r_mean"]
        self.assertLess(self.plan["d"], 2.0 * r_hat)
        self.assertGreater(self.plan["d"], 0.4 * r_hat)

    def test_directional_is_harder(self):
        """问题四：第二点还必须落在覆盖半平面内，探测概率不会更高。"""
        plan_dir = policy_mod.select_second_point(
            CFG, (500.0, 300.0), 30.0, rng=np.random.default_rng(7),
            n_mc=400, beta_step_deg=10.0, d_step_m=200.0, directional=True)
        self.assertLessEqual(plan_dir["pi"] + 1e-9, self.plan["pi"] + 0.2)


class TestOrthogonalityLaw(unittest.TestCase):
    def test_d_equals_2sqrt2_delta_r(self):
        """交会角 90° 时 D = 2√2·δ·r（问题一内核 + 问题二几何的交叉验证）。"""
        for r in (400.0, 800.0):
            P1 = (0.0, 0.0)
            d = r / math.cos(math.radians(45.0))
            P2 = (d * math.cos(math.radians(45.0)), d * math.sin(math.radians(45.0)))
            S = (r, 0.0)
            th2 = math.degrees(math.atan2(S[1] - P2[1], S[0] - P2[0]))
            D = policy_mod.batch_region_diameter(P1, 0.0, P2, np.array([th2]), DELTA)[0]
            expect = 2.0 * math.sqrt(2.0) * math.radians(DELTA) * r
            self.assertLess(abs(D / expect - 1.0), 2e-3)


if __name__ == "__main__":
    unittest.main()
