"""官方演练战果看板（只读，不改动任何日志）。

用途：`.jlog` 是加密密封的（AES-256-GCM + RSA-OAEP 包 DEK），正文读不出；但我们自己在
`cumcm2026b/runs/*-live*/trace.jsonl` 里完整记录了官方每个响应的原始字段，本脚本据此给出
**真实战果**与**官方时间口径**——第四问优化的目标函数就在这里。

    python paper/jlog_analyze.py                 # 汇总全部 live 会话
    python paper/jlog_analyze.py --session 020753   # 细看某一局（含逐频道明细）

判读要点（2026-09-12 实测）：
  · 时间不是窗口而是累加：/enter 返回 max_real_duration_s=1200、max_virtual_duration_s=360000，
    实测每局真实耗时 1.7~8.5 s、虚拟时间 3300~8058 s，12 局全部以我们自己的停止规则结束；
  · 战果口径：clear_result == "success" 的条数 = 真实清除的源数；
  · 典型损失：清除空挥率约 62%（±1° 在 1000~1300 m 处即 17~23 m 横向误差，与 20 m 清除半径同量级）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def load(session: str):
    ev = []
    path = os.path.join(RUNS, session, "trace.jsonl")
    if not os.path.exists(path):
        return ev
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
    return ev


def sessions() -> list:
    if not os.path.isdir(RUNS):
        return []
    return sorted(d for d in os.listdir(RUNS)
                  if "-live" in d and os.path.isdir(os.path.join(RUNS, d)))


def reason_of(session: str) -> str:
    rep = os.path.join(RUNS, session, "report.md")
    if not os.path.exists(rep):
        return "-"
    for line in open(rep, encoding="utf-8"):
        if "ended_reason" in line:
            return line.split("|")[2].strip()
    return "-"


def summarize(session: str) -> dict:
    ev = load(session)
    meas = n_dir = n_clear = n_ok = n_miss = 0
    ch_dir, ch_probe = set(), set()
    vts, walls = [], []
    for e in ev:
        if isinstance(e.get("wall_ms"), int):
            walls.append(e["wall_ms"])
        k = e.get("kind")
        if k == "measure":
            meas += 1
            ch = e.get("channel")
            if ch is not None:
                ch_probe.add(ch)
            if (e.get("result") or "") == "direction":
                n_dir += 1
                if ch is not None:
                    ch_dir.add(ch)
        if isinstance(e.get("vt"), (int, float)):
            vts.append(e["vt"])
        if k == "http" and e.get("path") == "/clear":
            n_clear += 1
            res = (e.get("body") or {}).get("clear_result")
            if res == "success":
                n_ok += 1
            elif res:
                n_miss += 1
    return {"session": session, "meas": meas, "dir": n_dir, "ch_probe": len(ch_probe),
            "ch_dir": len(ch_dir), "clear": n_clear, "success": n_ok, "miss": n_miss,
            "vt": vts[-1] if vts else -1.0,
            "wall_s": (max(walls) - min(walls)) / 1000.0 if walls else -1.0,
            "reason": reason_of(session)}


def enter_info(session: str) -> dict:
    for e in load(session):
        if e.get("kind") == "http" and e.get("path") == "/enter":
            return e.get("body") or {}
    return {}


def per_channel(session: str) -> dict:
    out = collections.defaultdict(lambda: {"m": 0, "d": 0, "ok": 0, "miss": 0,
                                           "pts": [], "svd": []})
    for e in load(session):
        if e.get("kind") == "measure":
            ch = e.get("channel")
            if ch is None:
                continue
            r = out[ch]
            r["m"] += 1
            if e.get("pos"):
                r["pts"].append(tuple(e["pos"]))
            if (e.get("result") or "") == "direction":
                r["d"] += 1
                if e.get("svd") is not None:
                    r["svd"].append(e["svd"])
        if e.get("kind") == "http" and e.get("path") == "/clear":
            ch = e.get("channel")
            if ch is None:
                continue
            if (e.get("body") or {}).get("clear_result") == "success":
                out[ch]["ok"] += 1
            else:
                out[ch]["miss"] += 1
    return out


def spread(pts) -> float:
    best = 0.0
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            best = max(best, math.hypot(a[0] - b[0], a[1] - b[1]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="会话目录名片段，细看该局")
    args = ap.parse_args()

    rows = [summarize(s) for s in sessions()]
    if not rows:
        print("没有找到 live 会话（runs/*-live*）")
        return 1
    print("%-24s %5s %5s %6s %5s %5s %5s %8s %7s %s"
          % ("session", "meas", "dir", "chProbe", "chDir", "succ", "miss", "vt_end", "wall_s", "reason"))
    for r in rows:
        print("%-24s %5d %5d %6d %5d %5d %5d %8.0f %7.1f %s"
              % (r["session"][:24], r["meas"], r["dir"], r["ch_probe"], r["ch_dir"],
                 r["success"], r["miss"], r["vt"], r["wall_s"], r["reason"]))
    n = len(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("meas", "dir", "clear", "success", "miss")}
    print("\n合计 %d 局：测量 %d（示向度 %d，%.1f%% 命中）、清除 %d（成功 %d，空挥 %d，%.1f%% 空挥）"
          % (n, tot["meas"], tot["dir"], 100.0 * tot["dir"] / max(tot["meas"], 1),
             tot["clear"], tot["success"], tot["miss"],
             100.0 * tot["miss"] / max(tot["clear"], 1)))
    print("平均每局：清除 %.1f 个源、虚拟时间 %.0f s、真实耗时 %.1f s"
          % (tot["success"] / n, sum(r["vt"] for r in rows) / n,
             sum(r["wall_s"] for r in rows) / n))

    sel = args.session
    if sel:
        cand = [r["session"] for r in rows if sel in r["session"]]
        if not cand:
            print("\n没有匹配的会话：%s" % sel)
            return 1
        s = cand[-1]
        print("\n=== 细看 %s ===" % s)
        info = enter_info(s)
        if info:
            print("官方时间口径：max_real=%s s, max_virtual=%s s, remaining_real=%s s"
                  % (info.get("max_real_duration_s"), info.get("max_virtual_duration_s"),
                     info.get("remaining_real_duration_s")))
        pc = per_channel(s)
        print("%-5s %4s %4s %5s %5s %9s  %s" % ("ch", "meas", "dir", "ok", "miss", "spread_m", "first svd"))
        for ch in sorted(pc):
            r = pc[ch]
            print("ch%-3s %4d %4d %5d %5d %9.0f  %s"
                  % (ch, r["m"], r["d"], r["ok"], r["miss"], spread(r["pts"]),
                     [round(v, 2) for v in r["svd"][:3]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
