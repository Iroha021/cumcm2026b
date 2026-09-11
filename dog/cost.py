"""虚拟时间与测向机状态镜像 —— 严格对齐附件 1 表 2 与附件 2 第 4 节。

规则（逐条实现，任何一条错了整盘策略都会偏）：
    /measure 总耗时 = 移动耗时 + 切换频道耗时 + 5
    /clear   总耗时 = 移动耗时 + (3 未发现 | 5 成功)，且不换频道、不改变当前频道
    移动耗时 = 上次合法动作位置到本次位置的直线距离 / 5 m/s
    切换频道耗时 = 1 秒，仅当合法 /measure 的 channel 与"当前频道"不同
    当前频道只由合法的 /measure 更新（/clear 永不更新）
    /enter、/exit 不推进虚拟时钟
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

Point = Tuple[float, float]


class CostModel(object):
    def __init__(self,
                 speed_mps: float = 5.0,
                 detect_s: float = 5.0,
                 switch_s: float = 1.0,
                 optical_s: float = 3.0,
                 clear_s: float = 5.0,
                 start_pos: Point = (0.0, 0.0),
                 start_channel: int = 1,
                 mismatch_tol_s: float = 1e-3):
        self.speed = float(speed_mps)
        self.detect_s = float(detect_s)
        self.switch_s = float(switch_s)
        self.optical_s = float(optical_s)
        self.clear_s = float(clear_s)
        self.position: Point = (float(start_pos[0]), float(start_pos[1]))
        self.channel: int = int(start_channel)
        self.virtual_time_s: float = 0.0
        self.tol = mismatch_tol_s
        self.mismatches: list = []

    # ------------------------------------------------------------ 代价预测
    def move_time(self, pos: Point) -> float:
        return math.hypot(pos[0] - self.position[0], pos[1] - self.position[1]) / self.speed

    def predict_measure(self, pos: Point, channel: int) -> Dict[str, float]:
        mv = self.move_time(pos)
        sw = 0.0 if int(channel) == self.channel else self.switch_s
        return {"move_s": mv, "switch_s": sw, "action_s": self.detect_s,
                "total_s": mv + sw + self.detect_s}

    def predict_clear(self, pos: Point, channel: int, found: Optional[bool] = None) -> Dict[str, float]:
        mv = self.move_time(pos)
        act = self.clear_s if found else self.optical_s
        return {"move_s": mv, "switch_s": 0.0, "action_s": act, "total_s": mv + act,
                "found": found}

    # ------------------------------------------------------------ 状态推进
    def on_measure_accepted(self, pos: Point, channel: int, virtual_time_s: float,
                            predicted_total: Optional[float] = None) -> None:
        self._apply(pos, virtual_time_s, predicted_total, "measure")
        self.channel = int(channel)          # 只有 measure 更新当前频道

    def on_clear_accepted(self, pos: Point, virtual_time_s: float,
                          predicted_total: Optional[float] = None) -> None:
        self._apply(pos, virtual_time_s, predicted_total, "clear")

    def on_enter(self, virtual_time_s: float) -> None:
        self.virtual_time_s = float(virtual_time_s)

    def _apply(self, pos: Point, virtual_time_s: float, predicted_total: Optional[float],
               kind: str) -> None:
        observed = float(virtual_time_s) - self.virtual_time_s
        if predicted_total is not None and abs(observed - predicted_total) > self.tol:
            self.mismatches.append({
                "kind": kind, "expected": predicted_total, "observed": observed,
                "at_pos": pos, "channel": self.channel,
            })
        self.position = (float(pos[0]), float(pos[1]))
        self.virtual_time_s = float(virtual_time_s)

    # ---------------------------------------------------------------- 工具
    def snapshot(self) -> Dict[str, object]:
        return {"pos": self.position, "channel": self.channel,
                "virtual_time_s": round(self.virtual_time_s, 6),
                "mismatches": len(self.mismatches)}
