"""Q4 诊断：每个真源的「命运」分解（不参与实时路径，只在 paper/ 下跑分析）。

回答一个问题：Q4 的清除比例损失到底丢在哪一段？
    A 已清除                        cleared
    B 测到过示向度、但始终没清掉      detected_not_cleared
    C 探过该频道、但一次都没测到      probed_not_detected
    D 从未探测过该频道                never_touched
并按 全向 / 定向 交叉统计，从而区分「遮挡盲区（几何）」与「清除失败（算法）」两类损失。

    python paper/q4_fate.py --cases 20 --scenario q4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                                  # noqa: E402

from dog import shadow as shadow_mod                                                # noqa: E402
from run import build, load_config                                                  # noqa: E402


class Recorder(object):
    """包一层 transport：记录每次 /measure 的位置与频道（用于判断"该测到却没测到"）。"""

    def __init__(self, inner):
        self.inner = inner
        self.probes = []
        self.clears = []            # [(channel, (x, y))] 每次 /clear 尝试点

    def __call__(self, path, payload):
        st, body = self.inner(path, payload)
        pos = payload.get("position") or {}
        ch = payload.get("channel")
        ok = st == 200 and body.get("accepted")
        if ok and "x" in pos and ch is not None:
            if path.endswith("/measure"):
                self.probes.append((int(ch), (float(pos["x"]), float(pos["y"]))))
            elif path.endswith("/clear"):
                self.clears.append((int(ch), (float(pos["x"]), float(pos["y"]))))
        return st, body
        return st, body


def classify(sim, world, rec):
    rows = []
    for s in sim.sources:
        b = world.beliefs.get(s.channel)
        n_bear = len(b.bearings) if b is not None else 0
        n_probe = getattr(b, "n_probes", 0) if b is not None else 0
        if s.cleared:
            fate = "A_cleared"
        elif n_bear > 0:
            fate = "B_detected_not_cleared"
        elif n_probe > 0:
            fate = "C_probed_not_detected"
        else:
            fate = "D_never_touched"

        d_min = float("inf")
        in_range = 0
        front = 0
        for ch, p in rec.probes:
            if ch != s.channel:
                continue
            d = math.hypot(p[0] - s.pos[0], p[1] - s.pos[1])
            d_min = min(d_min, d)
            if d <= s.R:
                in_range += 1
                # 覆盖判据：接收机相对源的方向 与 源朝向 α 的夹角 < 90°。
                # 源→接收机方向 = atan2(p - S)；探测点→源方向 = atan2(S - p)，两者差 180°。
                to_src = math.atan2(s.pos[1] - p[1], s.pos[0] - p[0])
                if s.omni or math.cos(math.radians(s.alpha) - (to_src + math.pi)) > 0.0:
                    front += 1

        n_clear = 0
        c_min = float("inf")
        for ch, p in rec.clears:
            if ch != s.channel:
                continue
            n_clear += 1
            c_min = min(c_min, math.hypot(p[0] - s.pos[0], p[1] - s.pos[1]))

        rows.append({"channel": s.channel, "omni": bool(s.omni), "R": round(s.R, 1),
                     "alpha": round(math.degrees(s.alpha), 1), "fate": fate,
                     "n_bearings": n_bear, "n_probes": n_probe,
                     "d_min_m": None if d_min == float("inf") else round(d_min, 1),
                     "n_probe_in_R": in_range, "n_probe_in_R_front": front,
                     "n_clear": n_clear,
                     "clear_min_m": None if c_min == float("inf") else round(c_min, 1),
                     "cleared": bool(s.cleared)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenario", default="q4", choices=["q3", "q4"])
    ap.add_argument("--sources", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    allrows = []
    per_case = []
    for i in range(args.cases):
        seed = args.seed + i
        sim = shadow_mod.ShadowSim(cfg, mode=args.scenario, seed=seed, n_sources=args.sources)
        rec = Recorder(sim.post)
        runner, world, _ = build(cfg, rec, "dryrun", args.scenario, seed, None)
        stats = runner.run(enter_wait_s=1.0)
        rows = classify(sim, world, rec)
        for r in rows:
            r["seed"] = seed
        allrows.extend(rows)
        n = len(rows)
        cl = sum(1 for r in rows if r["fate"] == "A_cleared")
        per_case.append({"seed": seed, "n": n, "cleared": cl,
                         "recall": cl / n if n else 0.0,
                         "vt": stats["virtual_time_s"],
                         "reason": stats["ended_reason"],
                         "probes": len(rec.probes)})
        print("[case %2d] seed=%d n=%d recall=%.3f vt=%.0f reason=%s"
              % (i + 1, seed, n, per_case[-1]["recall"], stats["virtual_time_s"],
                 stats["ended_reason"]), flush=True)

    print("\n=== 命运分解（%d 源 / %d 局 %s） ===" % (len(allrows), args.cases, args.scenario))
    order = ["A_cleared", "B_detected_not_cleared", "C_probed_not_detected", "D_never_touched"]
    for omni_tag, sel in (("全部", allrows),
                          ("全向源", [r for r in allrows if r["omni"]]),
                          ("定向源", [r for r in allrows if not r["omni"]])):
        if not sel:
            continue
        parts = []
        for f in order:
            c = sum(1 for r in sel if r["fate"] == f)
            parts.append("%s=%d(%.1f%%)" % (f.split("_", 1)[1], c, 100.0 * c / len(sel)))
        print("  %-6s n=%-4d %s" % (omni_tag, len(sel), "  ".join(parts)))

    bad = [r for r in allrows if not r["omni"] and r["fate"] in
           ("C_probed_not_detected", "D_never_touched") and r["n_probe_in_R_front"] > 0]
    print("\n  可疑（探到过覆盖区正面却没测到，应为 0）：%d" % len(bad))
    for r in bad[:10]:
        print("    seed=%d ch=%d R=%.0f alpha=%.1f d_min=%s in_R_front=%d"
              % (r["seed"], r["channel"], r["R"], r["alpha"], r["d_min_m"],
                 r["n_probe_in_R_front"]))

    miss = [r for r in allrows if r["fate"] in ("C_probed_not_detected", "D_never_touched")
            and r["d_min_m"] is not None]
    if miss:
        print("  未测到源的最近探测距离中位数: %.0f m（全体 R 中位数 %.0f m）"
              % (float(np.median([r["d_min_m"] for r in miss])),
                 float(np.median([r["R"] for r in allrows]))))
        blind = [r for r in miss if r["n_probe_in_R"] > 0]
        print("  其中「进入过 R 内但全在盲区」的源: %d / %d" % (len(blind), len(miss)))

    out = args.out or os.path.join(ROOT, "runs", "q4fate-%d.json" % args.cases)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"per_case": per_case, "rows": allrows}, fh, ensure_ascii=False, indent=1)
    print("明细已写入: %s" % out)
    print("  平均清除比例: %.4f  平均虚拟时间: %.1f s"
          % (float(np.mean([c["recall"] for c in per_case])),
             float(np.mean([c["vt"] for c in per_case]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
