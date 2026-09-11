"""行为日志：模拟器不提供日志，指令序列与响应必须由程序自己记录。

输出：runs/<run_id>/trace.jsonl  逐事件一行 JSON
      runs/<run_id>/report.md    运行结束后的汇总（对应题目表 1 的三项统计）
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class RunLogger(object):
    def __init__(self, runs_dir: str, tag: str = "", case_code: str = "", echo: bool = True):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = stamp + ("-" + tag if tag else "")
        self.run_id = run_id
        self.dir = os.path.abspath(os.path.join(runs_dir, run_id))
        os.makedirs(self.dir, exist_ok=True)
        self.case_code = case_code
        self.echo = echo
        self.seq = 0
        self._fh = open(os.path.join(self.dir, "trace.jsonl"), "w", encoding="utf-8")
        self.summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 事件
    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        self.seq += 1
        rec = {
            "seq": self.seq,
            "wall_ms": int(time.time() * 1000),
            "kind": kind,
        }
        rec.update(fields)
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        if self.echo:
            print(_fmt(rec), flush=True)
        return rec

    # ---------------------------------------------------------------- 汇总
    def finish(self, **fields: Any) -> None:
        self.summary.update(fields)
        self.summary["run_id"] = self.run_id
        self.summary["case_code"] = self.case_code
        with open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8") as fh:
            fh.write("# 运行报告 %s\n\n" % self.run_id)
            fh.write("| 项目 | 数值 |\n|---|---|\n")
            for k, v in self.summary.items():
                fh.write("| %s | %s |\n" % (k, _num(v)))
            fh.write("\n日志文件：`trace.jsonl`（共 %d 条事件）\n" % self.seq)
        self._fh.close()
        if self.echo:
            print("[report] %s" % json.dumps(self.summary, ensure_ascii=False, default=str), flush=True)


def _num(v: Any) -> str:
    if isinstance(v, float):
        return ("%.3f" % v).rstrip("0").rstrip(".")
    return str(v)


def _fmt(rec: Dict[str, Any]) -> str:
    """控制台摘要（保持 ASCII 前缀，便于重定向查看）。"""
    kind = rec.get("kind", "?")
    keys = ["pos", "channel", "result", "svd", "vt", "dt_vt", "dt_real", "reason", "cost_pred"]
    parts = []
    for k in keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, float):
                v = round(v, 3)
            parts.append("%s=%s" % (k, v))
    return "[%s] %s" % (kind, " ".join(parts))
