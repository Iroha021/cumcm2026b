"""影子模拟器（物理 + 计时），可作为 transport 注入 SimClient 用于 dryrun/调参。

只实现与策略决策相关的部分：目标区几何、源的真值分布、探测/清除规则、
虚拟时钟。协议层只做必要校验（字段白名单、request_id 幂等、取值范围），
用于尽早暴露我方 bug。

**输出仅用于相对比较与鲁棒性筛选，绝对性能一律以官方模拟器为准。**
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import geom

Point = Tuple[float, float]


class Source(object):
    __slots__ = ("channel", "pos", "R", "omni", "alpha", "cleared")

    def __init__(self, channel, pos, R, omni, alpha):
        self.channel = int(channel)
        self.pos = (float(pos[0]), float(pos[1]))
        self.R = float(R)
        self.omni = bool(omni)
        self.alpha = float(alpha)
        self.cleared = False


class ShadowSim(object):
    def __init__(self, cfg: Dict[str, Any], mode: str = "q3",
                 seed: Optional[int] = None, n_sources: Optional[int] = None,
                 directional_ratio: float = 0.5):
        self.cfg = cfg
        self.env = cfg["env"]
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.arena_r = float(self.env["arena_radius_m"])
        self.delta = float(self.env["bearing_error_deg"])
        self.speed = float(self.env["speed_mps"])
        self.dt_detect = float(self.env["detect_s"])
        self.dt_switch = float(self.env["switch_s"])
        self.dt_optical = float(self.env["optical_s"])
        self.dt_clear = float(self.env["clear_s"])
        self.near_r = float(self.env["near_radius_m"])
        self.clear_r = float(self.env["clear_radius_m"])
        self.r_min = float(self.env["recv_radius_min_m"])
        self.r_max = float(self.env["recv_radius_max_m"])

        self.sources: List[Source] = []
        self._generate(n_sources, directional_ratio)
        self.reset_runtime()

    # ------------------------------------------------------------------ 生成
    def _generate(self, n_sources: Optional[int], directional_ratio: float) -> None:
        if n_sources is None:
            lo = int(self.env["sources_min"])
            hi = int(self.env["sources_max"])
            n = int(self.rng.integers(lo, hi + 1))
        else:
            n = int(n_sources)
        ch_lo, ch_hi = int(self.env["channel_min"]), int(self.env["channel_max"])
        chans = self.rng.choice(np.arange(ch_lo, ch_hi + 1), size=n, replace=False)
        u = self.rng.random(n)
        rad = self.arena_r * np.sqrt(u)
        ang = self.rng.random(n) * 2.0 * math.pi
        for i in range(n):
            omni = True if self.mode == "q3" else (self.rng.random() >= directional_ratio)
            self.sources.append(Source(
                chans[i], (rad[i] * math.cos(ang[i]), rad[i] * math.sin(ang[i])),
                self.rng.uniform(self.r_min, self.r_max), omni,
                self.rng.uniform(0.0, 360.0)))
        self.sources.sort(key=lambda s: s.channel)

    def reset_runtime(self) -> None:
        self.position: Point = (0.0, 0.0)
        self.channel: int = int(self.env["channel_min"])
        self.virtual_time_s = 0.0
        self.entered = False
        self.exited = False
        self._idem: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self.n_requests = 0
        for s in self.sources:
            s.cleared = False

    def ground_truth(self) -> Dict[str, Any]:
        return {"n": len(self.sources),
                "directional": sum(1 for s in self.sources if not s.omni),
                "sources": [{"channel": s.channel, "pos": s.pos, "R": s.R,
                             "omni": s.omni, "alpha": s.alpha, "cleared": s.cleared}
                            for s in self.sources]}

    # ------------------------------------------------------------------ 工具
    def _bias(self, pos: Point) -> float:
        """位置固定的系统偏差（题面：同一地点重复检测误差不变）。"""
        key = "%.1f|%.1f" % (round(pos[0], 1), round(pos[1], 1))
        h = hashlib.sha256(key.encode("ascii", "replace")).digest()
        u = int.from_bytes(h[:4], "big") / float(1 << 32)
        return (u * 2.0 - 1.0) * self.delta

    def _source_of(self, channel: int) -> Optional[Source]:
        for s in self.sources:
            if s.channel == int(channel) and not s.cleared:
                return s
        return None

    def _detectable(self, s: Source, pos: Point) -> bool:
        if math.hypot(pos[0] - s.pos[0], pos[1] - s.pos[1]) > s.R:
            return False
        if s.omni:
            return True
        u = geom.unit(s.alpha)
        return ((pos[0] - s.pos[0]) * u[0] + (pos[1] - s.pos[1]) * u[1]) >= 0.0

    def _move_cost(self, pos: Point) -> float:
        return math.hypot(pos[0] - self.position[0], pos[1] - self.position[1]) / self.speed

    # --------------------------------------------------------------- 传输层
    def post(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        self.n_requests += 1
        rid = payload.get("request_id")
        if rid is not None and rid in self._idem:
            prev = self._idem[rid]
            if prev[1].get("_payload") == payload:
                body = dict(prev[1])
                body.pop("_payload", None)
                return prev[0], body
            return 409, self._err_body("request_id reused with different content")
        status, body = self._handle(path, payload)
        if rid is not None and status == 200 and body.get("accepted") is True:
            stored = dict(body)
            stored["_payload"] = payload
            self._idem[rid] = (status, stored)
        return status, body

    def _err_body(self, why: str) -> Dict[str, Any]:
        return {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0.0,
                "error": why}

    def _unknown_fields(self, payload: Dict[str, Any], allowed: set) -> List[str]:
        return [k for k in payload.keys() if k not in allowed]

    def _handle(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        base_allowed = {"arena_id", "robot_id", "request_id"}
        if path == "/enter" or path == "/exit":
            allowed = base_allowed
        elif path in ("/measure", "/clear"):
            allowed = base_allowed | {"position", "channel"}
        else:
            return 404, self._err_body("unknown path")
        if not isinstance(payload, dict):
            return 400, self._err_body("body must be a JSON object")
        if not base_allowed.issubset(payload.keys()):
            return 400, self._err_body("missing base field")
        if payload.get("arena_id") != "default":
            return 200, self._err_body("arena_id mismatch")
        if payload.get("robot_id") != self.cfg["team_id"]:
            return 200, self._err_body("robot_id mismatch")
        extra = self._unknown_fields(payload, allowed)
        if extra:
            return 200, self._err_body("unknown fields: %s" % extra)
        ch_lo, ch_hi = int(self.env["channel_min"]), int(self.env["channel_max"])
        if path in ("/measure", "/clear"):
            pos = payload.get("position")
            if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
                return 400, self._err_body("bad position")
            if set(pos.keys()) - {"x", "y"}:
                return 200, self._err_body("unknown position field")
            x, y = float(pos["x"]), float(pos["y"])
            if not (math.isfinite(x) and math.isfinite(y)) or max(abs(x), abs(y)) > 2.0e6:
                return 400, self._err_body("position out of range")
            ch = payload.get("channel")
            if not isinstance(ch, (int, float)) or float(ch) != int(ch):
                return 400, self._err_body("channel must be integer")
            ch = int(ch)
            if not (ch_lo <= ch <= ch_hi):
                return 400, self._err_body("channel out of range")
        if path == "/enter":
            if self.entered:
                return 200, self._err_body("duplicate enter")
            self.entered = True
            return 200, {"accepted": True, "real_timestamp_ms": 0,
                         "virtual_time_s": round(self.virtual_time_s, 6),
                         "max_virtual_duration_s": 360000,
                         "max_real_duration_s": 1200,
                         "remaining_real_duration_s": 1200}
        if not self.entered or self.exited:
            return 200, self._err_body("not in a running test")
        if path == "/measure":
            x, y = float(payload["position"]["x"]), float(payload["position"]["y"])
            ch = int(payload["channel"])
            mv = self._move_cost((x, y))
            sw = 0.0 if ch == self.channel else self.dt_switch
            self.virtual_time_s += mv + sw + self.dt_detect
            self.position = (x, y)
            self.channel = ch
            return 200, self._measure_body((x, y), ch)
        if path == "/clear":
            x, y = float(payload["position"]["x"]), float(payload["position"]["y"])
            ch = int(payload["channel"])
            mv = self._move_cost((x, y))
            self.position = (x, y)
            s = self._source_of(ch)
            if s is not None and math.hypot(x - s.pos[0], y - s.pos[1]) <= self.clear_r:
                self.virtual_time_s += mv + self.dt_clear
                s.cleared = True
                return 200, {"accepted": True, "real_timestamp_ms": 0,
                             "virtual_time_s": round(self.virtual_time_s, 6),
                             "clear_result": "success"}
            self.virtual_time_s += mv + self.dt_optical
            return 200, {"accepted": True, "real_timestamp_ms": 0,
                         "virtual_time_s": round(self.virtual_time_s, 6),
                         "clear_result": "no_target_in_range"}
        if path == "/exit":
            self.exited = True
            return 200, {"accepted": True, "real_timestamp_ms": 0,
                         "virtual_time_s": round(self.virtual_time_s, 6),
                         "exit_reason": "user_exit"}
        return 404, self._err_body("unknown path")

    def _measure_body(self, pos: Point, channel: int) -> Dict[str, Any]:
        s = self._source_of(channel)
        body = {"accepted": True, "real_timestamp_ms": 0,
                "virtual_time_s": round(self.virtual_time_s, 6)}
        if s is None or not self._detectable(s, pos):
            body["measure_result"] = "no_signal"
            return body
        d = math.hypot(pos[0] - s.pos[0], pos[1] - s.pos[1])
        if d <= self.near_r:
            body["measure_result"] = "near"
            return body
        true_brg = math.degrees(math.atan2(s.pos[1] - pos[1], s.pos[0] - pos[0])) % 360.0
        body["measure_result"] = "direction"
        body["svd_deg"] = round((true_brg + self._bias(pos)) % 360.0, 2)
        return body
