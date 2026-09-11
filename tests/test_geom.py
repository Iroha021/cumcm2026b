"""问题一几何内核自检：闭式解互验、退化识别、覆盖判定、Jung 界。"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import geom  # noqa: E402

DELTA = 1.0


def _bearing(p, s):
    """从检测点 p 指向源 s 的真实示向度（度）。"""
    return math.degrees(math.atan2(s[1] - p[1], s[0] - p[0])) % 360.0


class TestRegion(unittest.TestCase):
    def test_closed_form_matches_clipping(self):
        """两站闭式四角（正弦定理）与 Sutherland–Hodgman 结果一致。

        注意：两站必须"看向同一个源"，否则区域为空（这是正确的物理含义）。
        """
        cases = [((800.0, 500.0), (0.0, 0.0), (-300.0, 1100.0)),
                 ((-400.0, 900.0), (600.0, 200.0), (-1200.0, -500.0)),
                 ((1200.0, -300.0), (-500.0, 400.0), (300.0, -1400.0)),
                 ((150.0, 60.0), (-900.0, -200.0), (700.0, 900.0))]
        for S, p1, p2 in cases:
            st = [(p1, _bearing(p1, S)), (p2, _bearing(p2, S))]
            reg = geom.region_from_bearings(st, DELTA)
            corners = geom.region_from_sector_intersections(st, DELTA)
            self.assertTrue(reg.valid, "region should be bounded and non-empty")
            self.assertEqual(len(corners), 4)
            for p in reg.vertices:
                self.assertLess(min(geom.dist(p, q) for q in corners), 1e-6)

    def test_orthogonality_law(self):
        """正交几何下 D = 2*sqrt(2)*delta*r（已解析验证）。"""
        for r in (400.0, 855.0, 1300.0):
            P1 = (0.0, 0.0)
            th1 = 0.0
            d = r / math.cos(math.radians(45.0))
            P2 = (d * math.cos(math.radians(45.0)), d * math.sin(math.radians(45.0)))
            S = (r, 0.0)
            th2 = math.degrees(math.atan2(S[1] - P2[1], S[0] - P2[0]))
            reg = geom.region_from_bearings([(P1, th1), (P2, th2)], DELTA)
            D = reg.diameter()
            expect = 2.0 * math.sqrt(2.0) * math.radians(DELTA) * r
            self.assertLess(abs(D / expect - 1.0), 2e-3)

    def test_collinear_is_unbounded(self):
        """第二点落在第一示向线上 ⇒ 区域无界（禁飞走廊）。"""
        reg = geom.region_from_bearings([((0.0, 0.0), 30.0), ((1000.0, 0.0), 30.0)], DELTA)
        self.assertFalse(reg.bounded)
        self.assertEqual(reg.diameter(), float("inf"))
        self.assertTrue(geom.direction_intervals_intersect([30.0, 30.0], DELTA))

    def test_corridor_case_still_bounded(self):
        """相向示向（0°/180°）不构成无界：区域是两扇形夹出的细长四边形。"""
        reg = geom.region_from_bearings([((0.0, 0.0), 0.0), ((1000.0, 0.0), 180.0)], DELTA)
        self.assertTrue(reg.valid)
        self.assertAlmostEqual(reg.diameter(), 1000.0, places=3)

    def test_empty_region(self):
        reg = geom.region_from_bearings([((0.0, 0.0), 180.0), ((1000.0, 0.0), 0.0)], DELTA)
        self.assertTrue(reg.empty)

    def test_diameter_circle_may_fail(self):
        """以直径为直径的圆不一定覆盖区域：近等边三角形即反例。"""
        tri = [(0.0, 0.0), (40.0, 0.0), (20.0, 34.64)]
        D, (a, b) = geom.diameter(tri)
        self.assertLess(D, 40.01)
        self.assertFalse(geom.covered_by_diameter_circle(tri, a, b))
        _, r_mec = geom.min_enclosing_circle(tri)
        self.assertLessEqual(r_mec, D / math.sqrt(3.0) + 1e-6)   # Jung 定理上界
        self.assertGreater(r_mec, D / 2.0)

    def test_centrally_symmetric_is_covered(self):
        """中心对称（平行四边形）区域必被直径圆覆盖。"""
        quad = [(-30.0, -10.0), (30.0, -10.0), (30.0, 10.0), (-30.0, 10.0)]
        D, (a, b) = geom.diameter(quad)
        self.assertTrue(geom.covered_by_diameter_circle(quad, a, b))

    def test_disc_constraint(self):
        """域约束：追加目标区圆盘后区域直径不会变大。"""
        st = [((0.0, 0.0), 30.0), ((-300.0, 1100.0), -28.6)]
        base = geom.region_from_bearings(st, DELTA)
        hps = geom.disc_halfplanes((0.0, 0.0), 300.0, segments=64)
        clipped = geom.region_from_bearings(st, DELTA, extra_halfplanes=hps)
        self.assertLessEqual(clipped.diameter(), base.diameter() + 1e-6)


if __name__ == "__main__":
    unittest.main()
