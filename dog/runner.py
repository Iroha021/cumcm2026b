"""串行主循环：一次只允许一个在途动作（附件 2 第 1.4/5.3 节强制要求）。

职责：
    1) 生命周期：等待接口就绪 → /enter → 循环 → /exit
    2) 预算：以 /enter 返回的 remaining_real_duration_s 为硬上界，留安全裕度
    3) 落地每次请求与响应，并核对"预测耗时 vs 实测虚拟时间增量"（发现规则理解错误）
    4) 汇总题目表 1 需要的三项统计
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .sim_client import ProtocolError, RejectedError, SimClient, SimError, TransportError


class Runner(object):
    def __init__(self, cfg: Dict[str, Any], client: SimClient, world, cost, policy, logger,
                 mode: str = "live"):
        self.cfg = cfg
        self.client = client
        self.world = world
        self.cost = cost
        self.policy = policy
        self.logger = logger
        self.mode = mode
        self.n_actions = 0
        self.ended_reason = None
        self.test_over = False

    # ------------------------------------------------------------------ 主流程
    def run(self, enter_wait_s: float = 240.0) -> Dict[str, Any]:
        budget_cfg = self.cfg["budget"]
        margin = float(budget_cfg.get("real_time_safety_margin_s", 25.0))
        max_actions = int(budget_cfg.get("max_actions", 4000))
        vt_limit = 360000.0

        t_enter = time.monotonic()
        enter = self.client.enter_wait(enter_wait_s)
        remaining = float(enter.extra.get("remaining_real_duration_s", 0.0))
        self.cost.on_enter(enter.virtual_time_s)
        if self.logger:
            self.logger.event("enter", vt=round(self.cost.virtual_time_s, 3),
                              remaining_real_s=remaining,
                              max_real_s=enter.extra.get("max_real_duration_s"),
                              max_virtual_s=enter.extra.get("max_virtual_duration_s"))
        if remaining <= 0:
            self.ended_reason = "no_remaining_time"
            return self._finish(t_enter)

        deadline = time.monotonic() + max(0.0, remaining - margin)

        while True:
            if self.cost.virtual_time_s >= vt_limit:
                self.ended_reason = "virtual_limit"
                break
            if self.n_actions >= max_actions:
                self.ended_reason = "action_cap"
                break
            if time.monotonic() >= deadline:
                self.ended_reason = "real_budget_exhausted"
                break
            stop = self.policy.should_stop()
            if stop:
                self.ended_reason = stop
                break

            action = self.policy.decide()
            if action.kind == "exit":
                self.ended_reason = action.reason
                break

            left = deadline - time.monotonic()
            if action.expected_cost_s > left:
                self.ended_reason = "insufficient_budget"
                break

            try:
                outcome = self._execute(action)
            except TransportError as exc:
                self.test_over = True
                self.ended_reason = "test_ended_transport"
                if self.logger:
                    self.logger.event("ended", reason=self.ended_reason, detail=str(exc)[:200])
                break
            except (ProtocolError, RejectedError) as exc:
                self.ended_reason = "protocol_error"
                if self.logger:
                    self.logger.event("error", reason=self.ended_reason,
                                      detail=str(exc)[:400], action=repr(action))
                break

            self._apply(action, outcome)
            self.policy.note(action, outcome)
            self.n_actions += 1

        if not self.test_over:
            try:
                ex = self.client.exit()
                if self.logger:
                    self.logger.event("exit", reason=ex.extra.get("exit_reason"),
                                      vt=round(self.cost.virtual_time_s, 3))
            except SimError as exc:
                if self.logger:
                    self.logger.event("exit_failed", detail=str(exc)[:200])
        return self._finish(t_enter)

    # ------------------------------------------------------------------ 执行
    def _execute(self, action) -> Any:
        if action.kind == "measure":
            return self.client.measure(action.position, action.channel)
        if action.kind == "clear":
            return self.client.clear(action.position, action.channel)
        raise SimError("unknown action kind: %s" % action.kind)

    def _apply(self, action, outcome) -> None:
        pos = action.position
        vt = outcome.virtual_time_s
        if action.kind == "measure":
            pred = self.cost.predict_measure(pos, action.channel)
            self.cost.on_measure_accepted(pos, action.channel, vt, pred["total_s"])
            result = outcome.extra.get("result")
            svd = outcome.extra.get("svd_deg")
            if result == "direction":
                self.world.observe("direction", pos, action.channel, svd_deg=svd)
            elif result == "near":
                self.world.observe("near", pos, action.channel)
            else:
                self.world.observe("no_signal", pos, action.channel)
            if self.logger:
                self.logger.event("measure", pos=(round(pos[0], 2), round(pos[1], 2)),
                                  channel=action.channel, result=result, svd=svd,
                                  vt=round(vt, 3), cost_pred=round(pred["total_s"], 3),
                                  reason=action.reason, left_s=round(outcome.extra.get("left_s", 0.0), 1)
                                  if "left_s" in outcome.extra else None)
        elif action.kind == "clear":
            pred = self.cost.predict_clear(pos, action.channel,
                                           found=(outcome.extra.get("result") == "success"))
            self.cost.on_clear_accepted(pos, vt, pred["total_s"])
            ok = outcome.extra.get("result") == "success"
            self.world.observe("clear", pos, action.channel, clear_ok=ok)
            if self.logger:
                self.logger.event("clear", pos=(round(pos[0], 2), round(pos[1], 2)),
                                  channel=action.channel, result=outcome.extra.get("result"),
                                  vt=round(vt, 3), cost_pred=round(pred["total_s"], 3),
                                  reason=action.reason,
                                  cleared=len(self.world.cleared))

    # ------------------------------------------------------------------ 汇总
    def _finish(self, t_enter: float) -> Dict[str, Any]:
        real_s = time.monotonic() - t_enter
        cleared = len(self.world.cleared)
        vt = self.cost.virtual_time_s
        stats = {
            "run_id": getattr(self.logger, "run_id", None),
            "ended_reason": self.ended_reason,
            "cleared_sources": cleared,
            "actions": self.n_actions,
            "virtual_time_s": round(vt, 3),
            "program_real_time_s": round(real_s, 3),
            "avg_clear_virtual_s": round(vt / cleared, 3) if cleared else None,
            "cleared_channels": list(self.world.cleared),
            "channel_stats": self.world.stats(),
            "cost_model_mismatches": len(self.cost.mismatches),
            "virtual_per_real": round(vt / real_s, 2) if real_s > 0 else None,
        }
        if self.logger:
            self.logger.event("summary", **stats)
            self.logger.finish(**stats)
        return stats
