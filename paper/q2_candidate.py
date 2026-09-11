"""问题二：第二检测点选择与候选区域（离线高精度版 + 出图）。

    python run.py q2     # 或 python paper/q2_candidate.py

在线版本用的是同一函数（dog.policy.select_second_point），只是网格更粗、样本更少。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import policy as policy_mod  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def main(cfg=None):
    if cfg is None:
        from run import load_config
        cfg = load_config()
    os.makedirs(OUT, exist_ok=True)
    P1, th1 = (500.0, 300.0), 30.0
    plan = policy_mod.select_second_point(
        cfg, P1, th1, rng=np.random.default_rng(20260910),
        n_mc=1500, beta_step_deg=5.0, d_step_m=100.0)

    # 汇总表：固定 β=45°，扫 d
    rows = {}
    for r in plan["grid"]:
        rows.setdefault(r["beta"], []).append(r)
    table = sorted(rows.get(45.0, []), key=lambda r: r["d"])
    summary = {
        "P1": P1, "theta1_deg": th1,
        "prior_r_mean_m": round(plan["prior_r_mean"], 1),
        "best": {"P2": [round(plan["P2"][0], 1), round(plan["P2"][1], 1)],
                 "d_m": plan["d"], "beta_deg": plan["beta"],
                 "E_D_m": round(plan["E_D"], 2), "P90_D_m": round(plan["P90_D"], 2),
                 "P_detect": round(plan["pi"], 4), "crossing_angle_deg": round(plan["phi"], 1)},
        "candidate_region": plan["candidate_region"],
        "d_sweep_at_45deg": [{"d_m": r["d"], "E_D_m": round(r["E_D"], 2),
                              "P_detect": round(r["pi"], 4), "phi_deg": round(r["phi"], 1)}
                             for r in table],
    }
    with open(os.path.join(OUT, "q2_result.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    lines = ["# 问题二 结果", "",
             "- P1 = (%.0f, %.0f)，示向度 θ̂1 = %.1f°，源距离后验均值 r̂ = %.0f m"
             % (P1[0], P1[1], th1, summary["prior_r_mean_m"]), "",
             "## 推荐第二检测点", "",
             "| 项 | 数值 |", "|---|---|",
             "| P2 | (%.1f, %.1f) |" % (summary["best"]["P2"][0], summary["best"]["P2"][1]),
             "| 相对第一示向线的偏角 β* | %.1f° |" % summary["best"]["beta_deg"],
             "| 距离 d* | %.0f m |" % summary["best"]["d_m"],
             "| 期望定位区域直径 E[D] | %.2f m |" % summary["best"]["E_D_m"],
             "| 90 分位直径 P90[D] | %.2f m |" % summary["best"]["P90_D_m"],
             "| 第二点探测成功概率 | %.3f |" % summary["best"]["P_detect"],
             "| 平均交会角 | %.1f° |" % summary["best"]["crossing_angle_deg"], "",
             "## 候选区域（E[D] ≤ 1.2·min 且探测概率 ≥ 0.9）", "",
             "β ∈ [%.1f°, %.1f°]，d ∈ [%.0f, %.0f] m"
             % (summary["candidate_region"]["beta_min"], summary["candidate_region"]["beta_max"],
                summary["candidate_region"]["d_min"], summary["candidate_region"]["d_max"]), "",
             "## β = 45° 处沿 d 的扫描", "",
             "| d(m) | E[D](m) | 探测概率 | 交会角(°) |", "|---|---|---|---|"]
    for r in summary["d_sweep_at_45deg"]:
        lines.append("| %s | %s | %s | %s |" % (r["d_m"], r["E_D_m"], r["P_detect"], r["phi_deg"]))
    with open(os.path.join(OUT, "q2_result.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = plan["grid"]
        fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
        for axx, key, label in ((ax[0], "E_D", "E[D] (m)"), (ax[1], "pi", "P_detect")):
            xs = [g["beta"] for g in grid]
            ys = [g["d"] for g in grid]
            cs = [g[key] for g in grid]
            sc = axx.scatter(xs, ys, c=cs, s=12, cmap="viridis")
            axx.set_xlabel("beta (deg)")
            axx.set_ylabel("d (m)")
            axx.set_title(label)
            fig.colorbar(sc, ax=axx)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "q2_candidate_region.png"), dpi=160)
        print("figures written to", OUT)
    except Exception as exc:
        print("matplotlib unavailable, skipped figures:", exc)

    print(json.dumps({k: v for k, v in summary.items() if k != "d_sweep_at_45deg"},
                     ensure_ascii=False, indent=1))
    print("d-sweep at beta=45:", [(r["d_m"], r["E_D_m"], r["P_detect"]) for r in summary["d_sweep_at_45deg"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
