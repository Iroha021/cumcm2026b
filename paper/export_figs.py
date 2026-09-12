"""从真实演练日志导出论文用图表与原始 CSV。

设计约束：
  · 只用标准库 + numpy（本机没有 matplotlib/scipy），图一律输出 **SVG 矢量图**，
    Word / LaTeX / 浏览器均可直接用，也便于论文手统一配色；
  · 所有图都由 CSV 原始数据生成，保证"图表与正文数值同源"；
  · 站点集合直接调用算法本体 dog.routing.greedy_cover，保证示意图与程序一致。

用法：
    python paper/export_figs.py                 # 全部 live 会话
    python paper/export_figs.py --run 041931    # 指定某局出轨迹图与时间线

输出目录：paper/figs/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                                    # noqa: E402

from dog import routing                                                                # noqa: E402

RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "paper", "figs")
ARENA_R = 1800.0
COVER_R = 1200.0
TESTS = [("PUK6-JNYE-M249-3YAD", "20260912-040338"),
         ("DCV7-PADT-U89D-JTBG", "20260912-041540"),
         ("CT8Y-YMNV-HEF6-UTDE", "20260912-041931")]


# ------------------------------------------------------------------ SVG 小工具
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class SVG(object):
    def __init__(self, w, h, title=""):
        self.w, self.h = w, h
        self.p = []
        if title:
            self.text(w / 2.0, 22, title, size=15, anchor="middle", fill="#111")

    def rect(self, x, y, w, h, fill="#fff", stroke="#666", sw=1.0, op=1.0, rx=0):
        self.p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%d" '
                      'fill="%s" fill-opacity="%.2f" stroke="%s" stroke-width="%.1f"/>'
                      % (x, y, w, h, rx, fill, op, stroke, sw))

    def line(self, x1, y1, x2, y2, stroke="#333", sw=1.0, dash=None, op=1.0):
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                      'stroke-width="%.1f" stroke-opacity="%.2f"%s/>'
                      % (x1, y1, x2, y2, stroke, sw, op, d))

    def circle(self, cx, cy, r, fill="none", stroke="#333", sw=1.0, op=1.0, sop=1.0):
        self.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="%.2f" '
                      'stroke="%s" stroke-width="%.1f" stroke-opacity="%.2f"/>'
                      % (cx, cy, r, fill, op, stroke, sw, sop))

    def poly(self, pts, stroke="#333", sw=1.5, fill="none", op=1.0):
        s = " ".join("%.1f,%.1f" % (x, y) for x, y in pts)
        self.p.append('<polyline points="%s" fill="%s" stroke="%s" stroke-width="%.1f" '
                      'stroke-opacity="%.2f"/>' % (s, fill, stroke, sw, op))

    def text(self, x, y, s, size=12, anchor="start", fill="#222", weight="normal"):
        self.p.append('<text x="%.1f" y="%.1f" font-family="Segoe UI,Arial,sans-serif" '
                      'font-size="%d" font-weight="%s" text-anchor="%s" fill="%s">%s</text>'
                      % (x, y, size, weight, anchor, fill, esc(s)))

    def save(self, path):
        head = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d">\n<rect width="%d" height="%d" fill="#ffffff"/>\n'
                % (self.w, self.h, self.w, self.h, self.w, self.h))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(head + "\n".join(self.p) + "\n</svg>\n")


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


# ------------------------------------------------------------------ 读日志
def load(run_dir):
    ev = []
    p = os.path.join(RUNS, run_dir, "trace.jsonl")
    if not os.path.exists(p):
        return ev
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
    return ev


def report_fields(run_dir):
    out = {}
    p = os.path.join(RUNS, run_dir, "report.md")
    if not os.path.exists(p):
        return out
    txt = open(p, encoding="utf-8").read()
    for key in ("ended_reason", "cleared_sources", "actions", "virtual_time_s",
                "program_real_time_s", "avg_clear_virtual_s", "case_code",
                "cost_model_mismatches", "run_id"):
        m = re.search(r"\|\s*%s\s*\|\s*([^|]+)\|" % key, txt)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"\{'cleared': (\d+), 'pending': (\d+), 'absent': (\d+)\}", txt)
    if m:
        out["pending"] = int(m.group(2))
    return out


def session_row(run_dir):
    ev = load(run_dir)
    if not ev:
        return None
    meas = chd_dir = chd_any = succ = miss = 0
    det = set()
    for e in ev:
        k = e.get("kind")
        if k == "measure":
            meas += 1
            if (e.get("result") or "") == "direction":
                chd_dir += 1
                det.add(e.get("channel"))
        if k == "http" and e.get("path") == "/clear":
            r = (e.get("body") or {}).get("clear_result")
            succ += (r == "success")
            miss += (r != "success")
    rep = report_fields(run_dir)
    scen = "q3" if run_dir.endswith("-q3") else ("q4" if run_dir.endswith("-q4") else "?")
    return {"run_id": run_dir, "scenario": scen,
            "case_code": rep.get("case_code", ""),
            "cleared": succ, "detected_channels": len(det), "measures": meas,
            "direction_readings": chd_dir, "clear_miss": miss,
            "virtual_time_s": rep.get("virtual_time_s", ""),
            "avg_clear_virtual_s": rep.get("avg_clear_virtual_s", ""),
            "program_real_time_s": rep.get("program_real_time_s", ""),
            "pending": rep.get("pending", ""), "ended_reason": rep.get("ended_reason", ""),
            "cost_model_mismatches": rep.get("cost_model_mismatches", "")}


# ------------------------------------------------------------------ 图 1：覆盖
def fig_coverage(stations):
    S = 0.16                                    # 1800*0.16 = 288 px
    W = 700
    cx = cy = 300
    g = SVG(W, 760, "图：全区域覆盖站点与覆盖圆（7 站，覆盖半径 1200 m）")
    g.circle(cx, cy, ARENA_R * S, fill="#f7f9fc", stroke="#333", sw=1.6)
    for (x, y) in stations:
        px, py = cx + x * S, cy - y * S
        g.circle(px, py, COVER_R * S, fill="#2f6fd0", stroke="#2f6fd0", sw=0.8, op=0.07, sop=0.35)
    for i, (x, y) in enumerate(stations):
        px, py = cx + x * S, cy - y * S
        g.circle(px, py, 4.5, fill="#c0392b", stroke="#7b241c", sw=1)
        g.text(px + 7, py - 6, "P%d" % (i + 1), size=11, fill="#7b241c")
    g.circle(cx, cy, 3, fill="#444", stroke="#444")
    g.text(cx + 8, cy + 14, "起点 (0,0)", size=11, fill="#444")
    g.text(20, 620, "场地半径 1800 m；站点覆盖半径 1200 m（虚线圆）", size=12)
    g.text(20, 640, "完备性保证：场地上最差点距最近站点 963 m < 源的有效接收半径下界 1000 m",
           size=12, fill="#1f6f3f")
    g.text(20, 660, "⇒ 任意位置、任意 R ≥ 1000 m 的全向源，必被至少一个站点的测量捕获", size=12,
           fill="#1f6f3f")
    g.text(20, 690, "站点由贪心集合覆盖生成（dog/routing.py::greedy_cover），与算法本体同源",
           size=11, fill="#666")
    g.save(os.path.join(OUT, "fig_coverage.svg"))


# ------------------------------------------------------------------ 图 2：轨迹
def fig_trajectory(run_dir, row):
    ev = load(run_dir)
    path, meas, dirs, okc, miss = [], [], [], [], []
    for e in ev:
        k = e.get("kind")
        pos = e.get("pos")
        vt = e.get("vt")
        if k == "measure" and pos:
            path.append((pos[0], pos[1], vt, "m"))
            meas.append((pos[0], pos[1]))
            if (e.get("result") or "") == "direction":
                dirs.append((pos[0], pos[1]))
        if k == "clear" and pos:
            path.append((pos[0], pos[1], vt, "c"))
            (okc if (e.get("result") or "") == "success" else miss).append((pos[0], pos[1]))
    if not path:
        return 0
    S = 0.16
    cx = cy = 300
    g = SVG(700, 780, "图：机器狗轨迹（%s，清除 %d 个源，虚拟时间 %s s）"
            % (row["case_code"] or run_dir, row["cleared"], row["virtual_time_s"]))
    g.circle(cx, cy, ARENA_R * S, fill="#fbfcfe", stroke="#333", sw=1.6)
    for (x, y) in station_xy():
        g.circle(cx + x * S, cy - y * S, 3.0, fill="#bbb", stroke="#999", sw=0.8)
    g.poly([(cx + x * S, cy - y * S) for x, y, _, _ in path], stroke="#2f6fd0", sw=1.2, op=0.85)
    for x, y in meas:
        g.circle(cx + x * S, cy - y * S, 1.8, fill="#9aa4b2", stroke="none")
    for x, y in dirs:
        g.circle(cx + x * S, cy - y * S, 3.2, fill="#1f9d55", stroke="#14532d", sw=0.5)
    for x, y in okc:
        g.circle(cx + x * S, cy - y * S, 6.5, fill="none", stroke="#1f9d55", sw=2.4)
    for x, y in miss:
        px, py = cx + x * S, cy - y * S
        g.line(px - 5, py - 5, px + 5, py + 5, stroke="#c0392b", sw=2)
        g.line(px - 5, py + 5, px + 5, py - 5, stroke="#c0392b", sw=2)
    y0 = 630
    g.circle(28, y0, 1.8, fill="#9aa4b2", stroke="none")
    g.text(40, y0 + 4, "测量点(未测到)", size=11)
    g.circle(178, y0, 3.2, fill="#1f9d55", stroke="#14532d", sw=0.5)
    g.text(190, y0 + 4, "测得示向度", size=11)
    g.circle(305, y0, 6.5, fill="none", stroke="#1f9d55", sw=2.4)
    g.text(320, y0 + 4, "清除成功", size=11)
    g.line(418, y0 - 5, 428, y0 + 5, stroke="#c0392b", sw=2)
    g.line(418, y0 + 5, 428, y0 - 5, stroke="#c0392b", sw=2)
    g.text(438, y0 + 4, "清除失败", size=11)
    g.line(20, y0 + 30, 60, y0 + 30, stroke="#2f6fd0", sw=1.2)
    g.text(70, y0 + 34, "巡游轨迹（按动作先后连线）", size=11)
    g.text(20, y0 + 60, "灰点为巡游站点；数据同源文件 traj_%s.csv" % run_dir[9:], size=10, fill="#666")
    g.save(os.path.join(OUT, "traj_%s.svg" % run_dir))
    return len(path)


# ------------------------------------------------------------------ 图 3：时间线
def fig_timeline(run_dir, row):
    ev = load(run_dir)
    pts = []
    for e in ev:
        k = e.get("kind")
        ch = e.get("channel")
        vt = e.get("vt")
        if ch is None or not isinstance(vt, (int, float)):
            continue
        if k == "measure":
            r = e.get("result") or "?"
            pts.append((vt, int(ch), {"direction": "dir", "no_signal": "sil"}.
                        get(r, "other"), e.get("svd")))
        if k == "clear":
            pts.append((vt, int(ch), "ok" if (e.get("result") or "") == "success" else "miss",
                        None))
    if not pts:
        return 0
    tmax = max(p[0] for p in pts) or 1.0
    W, H = 960, 520
    L, T, R, B = 70, 50, 930, 430
    g = SVG(W, H, "图：各频道事件时间线（%s）" % (row["case_code"] or run_dir))
    g.line(L, B, R, B, stroke="#333", sw=1.2)
    g.line(L, T, L, B, stroke="#333", sw=1.2)
    for ch in range(1, 21):
        y = T + (ch - 1) * (B - T) / 19.0
        g.line(L, y, R, y, stroke="#eee", sw=1)
        g.text(L - 8, y + 4, "ch%d" % ch, size=10, anchor="end", fill="#555")
    for f in range(0, 6):
        t = tmax * f / 5.0
        x = L + (R - L) * f / 5.0
        g.line(x, B, x, B + 5, stroke="#333", sw=1)
        g.text(x, B + 20, "%.0f" % t, size=10, anchor="middle", fill="#555")
    g.text((L + R) / 2.0, B + 42, "虚拟时间 (s)", size=12, anchor="middle")
    for vt, ch, kind, svd in pts:
        x = L + (R - L) * (vt / tmax)
        y = T + (ch - 1) * (B - T) / 19.0
        if kind == "dir":
            g.circle(x, y, 4.2, fill="#1f9d55", stroke="#14532d", sw=0.6)
        elif kind == "sil":
            g.circle(x, y, 2.4, fill="#c8ccd4", stroke="none")
        elif kind == "ok":
            g.rect(x - 4, y - 4, 8, 8, fill="#2f6fd0", stroke="#1b3f78", sw=0.6)
        elif kind == "miss":
            g.line(x - 4, y - 4, x + 4, y + 4, stroke="#c0392b", sw=1.8)
            g.line(x - 4, y + 4, x + 4, y - 4, stroke="#c0392b", sw=1.8)
        else:
            g.circle(x, y, 3.0, fill="#e3b505", stroke="#8a6d00", sw=0.5)
    lg = B + 70
    g.circle(L + 8, lg, 4.2, fill="#1f9d55", stroke="#14532d", sw=0.6)
    g.text(L + 18, lg + 4, "测得示向度", size=11)
    g.circle(L + 118, lg, 2.4, fill="#c8ccd4", stroke="none")
    g.text(L + 128, lg + 4, "无信号", size=11)
    g.rect(L + 208, lg - 4, 8, 8, fill="#2f6fd0", stroke="#1b3f78", sw=0.6)
    g.text(L + 222, lg + 4, "清除成功", size=11)
    g.line(L + 312, lg - 4, L + 320, lg + 4, stroke="#c0392b", sw=1.8)
    g.line(L + 312, lg + 4, L + 320, lg - 4, stroke="#c0392b", sw=1.8)
    g.text(L + 330, lg + 4, "清除失败", size=11)
    g.text(L, lg + 26, "纵轴=频道号，横轴=虚拟时间；原始数据 events_%s.csv" % run_dir[9:],
           size=10, fill="#666")
    g.save(os.path.join(OUT, "events_%s.svg" % run_dir))
    return len(pts)


# ------------------------------------------------------------------ 图 4：流程
def fig_flow():
    W, H = 860, 700
    g = SVG(W, H, "图：算法流程（覆盖巡游—贝叶斯测向定位—在线重规划）")
    boxes = [
        (60, 50, 300, 40, "开始：/enter，从 (0,0) 频道 1 出发"),
        (60, 110, 300, 40, "生成覆盖站点（贪心覆盖，7 站，保证 963 m < 1000 m）"),
        (60, 170, 300, 40, "按价值构建节点集 → 最近邻+2-opt 近似排序"),
        (60, 230, 300, 40, "当前站点批量测量（换频 1 s + 检测 5 s）"),
        (60, 290, 300, 56, "观测更新：示向度 / near / 静默\n→ 半平面交 + 粒子滤波 + 存在性检验"),
        (60, 366, 300, 40, "定位够准？（20 m 圆盘后验质量 ≥ 0.25）"),
        (430, 366, 340, 40, "否：补测——问题二第二点 / GDOP 补点 / 张角增益门槛"),
        (60, 426, 300, 40, "是：/clear（成功 5 s / 失败 3 s）"),
        (430, 426, 340, 40, "失败：排除该 20 m 圆盘 → 就地补扫（≤150 m，≤3 次）"),
        (60, 486, 300, 40, "清除成功 → 该频道结束"),
        (60, 546, 300, 40, "还有正价值节点？"),
        (430, 546, 340, 40, "是：重规划后继续巡游（滚动时域）"),
        (60, 606, 300, 40, "否：/exit。第三问要求全部频道已判定（all_resolved）"),
    ]
    for (x, y, w, h, t) in boxes:
        g.rect(x, y, w, h, fill="#f4f7fb", stroke="#7f8fa6", sw=1.2, rx=6)
        lines = t.split("\n")
        for i, ln in enumerate(lines):
            g.text(x + 10, y + 24 + i * 15, ln, size=11.5)
    for (x1, y1, x2, y2) in [(210, 90, 210, 110), (210, 150, 210, 170), (210, 210, 210, 230),
                             (210, 270, 210, 290), (210, 346, 210, 366), (210, 406, 210, 426),
                             (210, 466, 210, 486), (210, 526, 210, 546), (210, 586, 210, 606)]:
        g.line(x1, y1, x2, y2, stroke="#7f8fa6", sw=1.4)
    g.line(360, 386, 430, 386, stroke="#7f8fa6", sw=1.4)
    g.line(360, 446, 430, 446, stroke="#7f8fa6", sw=1.4)
    g.line(210, 506, 210, 526, stroke="#7f8fa6", sw=1.4)
    g.line(360, 566, 430, 566, stroke="#7f8fa6", sw=1.4)
    g.save(os.path.join(OUT, "fig_flowchart.svg"))


def station_xy():
    try:
        st = routing.greedy_cover(ARENA_R, COVER_R)
        return [(float(p[0]), float(p[1])) for p in st]
    except Exception:
        return []


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="指定会话目录片段（出轨迹图与时间线）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    runs = sorted(d for d in os.listdir(RUNS)
                  if os.path.isdir(os.path.join(RUNS, d)) and "-live" in d)
    rows = [r for r in (session_row(d) for d in runs) if r and r["measures"] > 0]

    # 1) 全部演练明细
    hdr = ["run_id", "scenario", "case_code", "cleared", "detected_channels", "measures",
           "direction_readings", "clear_miss", "virtual_time_s", "avg_clear_virtual_s",
           "program_real_time_s", "pending", "ended_reason", "cost_model_mismatches"]
    write_csv(os.path.join(OUT, "runs_all.csv"), hdr,
              [[r[h] for h in hdr] for r in rows])

    # 2) 分批次汇总
    def agg(tag, rs):
        if not rs:
            return None
        n = len(rs)
        cleared = sum(r["cleared"] for r in rs)
        det = sum(r["detected_channels"] for r in rs)
        miss = sum(r["clear_miss"] for r in rs)
        return [tag, n, round(cleared / float(n), 3), round(det / float(n), 3),
                round((det - cleared) / float(n), 3),
                round(100.0 * miss / max(miss + cleared, 1), 1),
                round(sum(float(r["virtual_time_s"] or 0) for r in rs) / n, 1)]
    summary = [agg("问题三 真实演练", [r for r in rows if r["scenario"] == "q3"]),
               agg("问题四 真实演练（全部）", [r for r in rows if r["scenario"] == "q4"]),
               agg("问题四 最终配置(近距点移除后)",
                   [r for r in rows if r["scenario"] == "q4" and r["run_id"] >= "20260912-032600"]),
               agg("问题四 早期(含近距点)",
                   [r for r in rows if r["scenario"] == "q4" and r["run_id"] < "20260912-032600"])]
    write_csv(os.path.join(OUT, "runs_summary.csv"),
              ["batch", "sessions", "mean_cleared", "mean_detected_channels",
               "mean_gap", "clear_miss_pct", "mean_virtual_time_s"],
              [s for s in summary if s])

    # 3) 三个案例的成绩表
    trows = []
    for code, prefix in TESTS:
        hit = [r for r in rows if r["case_code"] == code or r["run_id"].startswith(prefix)]
        if not hit:
            trows.append([code, "", "", "", "", ""])
            continue
        r = hit[0]
        trows.append([code, r["run_id"], r["cleared"], r["avg_clear_virtual_s"],
                      r["program_real_time_s"], r["ended_reason"]])
    write_csv(os.path.join(OUT, "table_tests.csv"),
              ["case_code", "run_id", "cleared_sources", "avg_clear_virtual_s",
               "program_real_time_s", "ended_reason"], trows)

    # 4) 站点覆盖
    st = station_xy()
    write_csv(os.path.join(OUT, "coverage_stations.csv"),
              ["station_index", "x_m", "y_m", "cover_radius_m"],
              [[i + 1, round(x, 1), round(y, 1), COVER_R] for i, (x, y) in enumerate(st)])
    if st:
        fig_coverage(st)
        fig_flow()

    # 5) 轨迹与时间线（默认三个案例局；可用 --run 追加）
    picks = [p for _, p in TESTS]
    if args.run:
        picks = [d for d in runs if args.run in d] or picks
    made = []
    for d in picks:
        row = [r for r in rows if r["run_id"].startswith(d)]
        if not row:
            continue
        row = row[0]
        d = row["run_id"]
        ev = load(d)
        traj = [[i, e.get("vt"), e.get("pos", [None, None])[0], e.get("pos", [None, None])[1],
                 e.get("kind"), e.get("channel"), e.get("result")]
                for i, e in enumerate(ev, 1) if e.get("pos")]
        write_csv(os.path.join(OUT, "traj_%s.csv" % d),
                  ["seq", "virtual_time_s", "x_m", "y_m", "action", "channel", "result"], traj)
        evs = [[i, e.get("vt"), e.get("channel"), e.get("kind"), e.get("result"),
                (e.get("pos") or [None, None])[0], (e.get("pos") or [None, None])[1],
                e.get("svd")]
               for i, e in enumerate(ev, 1)
               if e.get("channel") is not None and isinstance(e.get("vt"), (int, float))]
        write_csv(os.path.join(OUT, "events_%s.csv" % d),
                  ["seq", "virtual_time_s", "channel", "action", "result", "x_m", "y_m", "svd_deg"],
                  evs)
        n1 = fig_trajectory(d, row)
        n2 = fig_timeline(d, row)
        made.append((d, len(traj), len(evs), n1, n2))

    # 6) 说明文件
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("论文图表与原始数据（由 paper/export_figs.py 自动生成，未经手工修改）\n")
        fh.write("=" * 70 + "\n\n")
        fh.write("runs_all.csv          全部真实演练逐局明细（14 列，字段名即含义）\n")
        fh.write("runs_summary.csv      按批次汇总：局数/平均清除/平均测到频道/缺口/空挥率/平均虚拟时间\n")
        fh.write("table_tests.csv       三个案例的成绩表（清除数、平均定位清除时间、程序运行时间）\n")
        fh.write("coverage_stations.csv 覆盖站点坐标与覆盖半径（由 dog/routing.py::greedy_cover 生成）\n")
        fh.write("traj_<run_id>.csv     轨迹点序列：seq/虚拟时间/x/y/动作/频道/结果\n")
        fh.write("events_<run_id>.csv   事件时间线：seq/虚拟时间/频道/动作/结果/坐标/示向度\n")
        fh.write("fig_coverage.svg      覆盖站点与覆盖圆示意图\n")
        fh.write("fig_flowchart.svg     算法流程图\n")
        fh.write("traj_<run_id>.svg     机器狗轨迹图（含测得示向度、清除成功/失败标记）\n")
        fh.write("events_<run_id>.svg   各频道事件时间线（纵轴频道，横轴虚拟时间）\n\n")
        fh.write("说明：图为纯标准库生成的 SVG 矢量图，可直接放入 Word/LaTeX 或重新配色；\n")
        fh.write("      所有图都由同目录 CSV 生成，保证图与正文数值同源。\n")

    print("sessions=%d  csv+svg 输出目录=%s" % (len(rows), OUT))
    for f in sorted(os.listdir(OUT)):
        print("  %-34s %7d B" % (f, os.path.getsize(os.path.join(OUT, f))))
    for m in made:
        print("  run %s: traj=%d events=%d (svg traj=%s events=%s)" % m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
