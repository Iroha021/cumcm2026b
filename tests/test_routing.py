"""寻路模块自检：GDOP 公式、120° 补点、巡游排序。"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import routing  # noqa: E402


class TestGdop(unittest.TestCase):
    def test_right_angle_value(self):
        """s1=(0,0)、s2=(2,0)、w=(1,1)：d1=d2=√2、交会角 90° ⇒ U=2。"""
        u = routing.gdop((0.0, 0.0), (2.0, 0.0), (1.0, 1.0))
        self.assertAlmostEqual(u, 2.0, places=9)

    def test_collinear_diverges(self):
        self.assertEqual(routing.gdop((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)), float("inf"))

    def test_best_pair_improves_with_new_point(self):
        """加入一个与已有观测张开大角的点后，最好一对的 GDOP 不应变差。"""
        w = (0.0, 0.0)
        obs = [(1000.0, 200.0)]
        u1 = routing.best_pair_gdop(obs, w)
        self.assertEqual(u1, float("inf"))               # 只有一条，无法交会
        obs2 = obs + [(-800.0, 600.0)]
        u2 = routing.best_pair_gdop(obs2, w)
        self.assertLess(u2, 2.0e6 + 1.0)

    def test_crossing_angle(self):
        self.assertAlmostEqual(routing.crossing_angle_deg((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
                               90.0, places=6)


class TestRefinePoint(unittest.TestCase):
    def test_picks_large_angle_side(self):
        """目标在原点，唯一观测点在 +x 方向 ⇒ 补测点应落在 ±120° 附近（而非 0°）。"""
        target = (0.0, 0.0)
        obs = [(1000.0, 0.0)]
        p = routing.choose_refine_point(target, obs, u_star=6.6e5, r_guaranteed=1000.0)
        self.assertIsNotNone(p)
        ang = math.degrees(math.atan2(p[1], p[0])) % 360.0
        d = min(abs(ang - 120.0), abs(ang - 240.0), abs(ang - 180.0))
        self.assertLess(d, 40.0, "补测点没有落在角多样性好的一侧：%.1f°" % ang)
        self.assertLessEqual(math.hypot(p[0] - target[0], p[1] - target[1]), 1000.0 + 1e-6)


class TestTour(unittest.TestCase):
    def test_matches_greedy_or_better(self):
        pts = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0),
               (500.0, 500.0), (200.0, 800.0)]
        start = (-500.0, -500.0)
        order, cost = routing.tour_order(pts, start)
        nn = routing.nearest_neighbor_order(pts, start)

        def cost_of(o):
            if not o:
                return 0.0
            total = math.dist(pts[o[0]], start)
            for a, b in zip(o, o[1:]):
                total += math.dist(pts[a], pts[b])
            return total

        self.assertLessEqual(cost, cost_of(nn) + 1e-6)
        self.assertEqual(sorted(order), list(range(len(pts))))

    def test_empty(self):
        self.assertEqual(routing.tour_order([], (0.0, 0.0)), ([], 0.0))


class TestConversion(unittest.TestCase):
    def test_round_trip(self):
        delta = math.radians(1.0)
        d = 40.0
        u = routing.gdop_from_region_diameter(d, delta)
        self.assertAlmostEqual(routing.region_diameter_from_gdop(u, delta), d, places=6)
        # r≈810 m 时 U*≈6.6e5，与 config 中的取值一致
        self.assertLess(abs(u - 6.6e5) / 6.6e5, 0.05)


if __name__ == "__main__":
    unittest.main()
