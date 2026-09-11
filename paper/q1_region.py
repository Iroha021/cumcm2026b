"""问题一：定位区域、直径、覆盖判定的离线计算与出图（不参与实时运行）。

只调用运行时内核 dog.geom，保证"论文里的算法"与"程序里的算法"是同一份实现。

    python run.py q1     # 或 python paper/q1_region.py
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import geom  # noqa: E402

DELTA = 1.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _outdir() -> str:
    os.makedirs(OUT, exist_ok=True)
    return OUT


def demo_region(n=6):
    """随机两站/多站构型：区域、直径、覆盖判定。"""
    import random
    rng = random.Random(20260910)
    rows = []
    for k in range(n):
        S = (rng.uniform(-1200, 1200), rng.uniform(-1200, 1200))
        stations = []
        for _ in range(rng.choice([2, 2, 3, 4])):
            P = (rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
            th = math.degrees(math.atan2(S[1] - P[1], S[0] - P[0])) % 360.0
            stations.append((P, th))
        reg = geom.region_from_bearings(stations, DELTA)
        if not reg.valid:
            rows.append({"n_stations": len(stations), "status": "unbounded/empty"})
            continue
        D, (a, b) = geom.diameter(reg.vertices)
        cov = geom.covered_by_diameter_circle(reg.vertices, a, b)
        mec_c, mec_r = geom.min_enclosing_circle(reg.vertices)
        rows.append({"n_stations": len(stations), "n_vertices": len(reg.vertices),
                     "diameter_m": round(D, 3), "covered_by_diameter_circle": cov,
                     "mec_radius_m": round(mec_r, 3),
                     "ratio_mec_over_halfD": round(mec_r / (D / 2.0), 4),
                     "jung_bound": round(2.0 / math.sqrt(3.0), 4)})
    return rows


def sweep_crossing_angle():
    """对称构型下"交会角 φ — 区域直径 D"的关系（正交最优的数值证据）。"""
    rows = []
    r = 1000.0
    for phi in (10, 15, 30, 45, 60, 75, 90, 105, 120, 150):
        ph = math.radians(phi)
        P1 = (-r, 0.0)
        P2 = (r * math.cos(ph), r * math.sin(ph))
        S = (0.0, 0.0)
        th1 = math.degrees(math.atan2(S[1] - P1[1], S[0] - P1[0]))
        th2 = math.degrees(math.atan2(S[1] - P2[1], S[0] - P2[0]))
        reg = geom.region_from_bearings([(P1, th1), (P2, th2)], DELTA)
        D = reg.diameter() if reg.valid else float("inf")
        rows.append({"phi_deg": phi, "diameter_m": round(D, 2) if D != float("inf") else None,
                     "D_sin_phi_over_2delta_r": round(D * math.sin(ph) / (2 * math.radians(DELTA) * r), 4)
                     if D != float("inf") else None})
    return rows


def counterexample():
    """"以直径为直径的圆能否覆盖定位区域"的反例与界。"""
    tri = [(0.0, 0.0), (41.32, 0.0), (20.66, 35.79)]        # 近等边三角形
    D, (a, b) = geom.diameter(tri)
    m = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    worst = max(geom.dist(m, p) for p in tri)
    mec_c, mec_r = geom.min_enclosing_circle(tri)
    return {
        "vertices": tri,
        "diameter_m": round(D, 3),
        "diameter_circle_radius_m": round(D / 2.0, 3),
        "required_radius_m": round(worst, 3),
        "required_over_D": round(worst / D, 4),
        "covered": geom.covered_by_diameter_circle(tri, a, b),
        "min_enclosing_circle_radius_m": round(mec_r, 3),
        "mec_over_D": round(mec_r / D, 4),
        "jung_bound_radius_over_D": round(1.0 / math.sqrt(3.0), 4),
        "excess_m": round(worst - D / 2.0, 3),
    }


def main(cfg=None):
    outdir = _outdir()
    res = {
        "crossing_angle_sweep": sweep_crossing_angle(),
        "coverage_counterexample": counterexample(),
        "random_regions": demo_region(),
    }
    path = os.path.join(outdir, "q1_result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    lines = ["# 问题一 结果", "", "## 1. 交会角 — 区域直径（r=1000 m, δ=1°）", "",
             "| φ(°) | 直径 D(m) | D·sinφ/(2δr) |", "|---|---|---|"]
    for r in res["crossing_angle_sweep"]:
        lines.append("| %s | %s | %s |" % (r["phi_deg"], r["diameter_m"], r["D_sin_phi_over_2delta_r"]))
    ce = res["coverage_counterexample"]
    lines += ["", "## 2. 覆盖性反例（近等边三角形）", "",
              "| 项 | 数值 |", "|---|---|",
              "| 区域直径 D | %.3f m |" % ce["diameter_m"],
              "| 直径圆半径 D/2 | %.3f m |" % ce["diameter_circle_radius_m"],
              "| 实际所需半径 | %.3f m |" % ce["required_radius_m"],
              "| 超出量 | %.3f m |" % ce["excess_m"],
              "| 能否覆盖 | %s |" % ce["covered"],
              "| 最小包围圆半径 | %.3f m (%.4f·D) |" % (ce["min_enclosing_circle_radius_m"], ce["mec_over_D"]),
              "| Jung 上界 | %.4f·D |" % ce["jung_bound_radius_over_D"], ""]
    with open(os.path.join(outdir, "q1_result.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["phi_deg"] for r in res["crossing_angle_sweep"]]
        ys = [r["diameter_m"] for r in res["crossing_angle_sweep"]]
        plt.figure(figsize=(5.2, 3.6))
        plt.plot(xs, ys, "o-")
        plt.axvline(90, ls="--", c="gray")
        plt.xlabel("crossing angle phi (deg)")
        plt.ylabel("region diameter D (m)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "q1_diameter_vs_phi.png"), dpi=160)
        print("figures written to", outdir)
    except Exception as exc:                     # matplotlib 非必需
        print("matplotlib unavailable, skipped figures:", exc)

    print(json.dumps(res["coverage_counterexample"], ensure_ascii=False, indent=1))
    print("crossing-angle sweep:", [(r["phi_deg"], r["diameter_m"]) for r in res["crossing_angle_sweep"]])
    print("outputs:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
