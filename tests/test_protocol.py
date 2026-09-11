"""协议与端到端自检（用影子模拟器，不接触官方模拟器）。"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog import shadow as shadow_mod  # noqa: E402
from dog.cost import CostModel  # noqa: E402
from dog.sim_client import ProtocolError, RejectedError, SimClient  # noqa: E402
from run import build, load_config  # noqa: E402

CFG = load_config()


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.sim = shadow_mod.ShadowSim(CFG, mode="q3", seed=1, n_sources=12)
        self.client = SimClient(CFG, logger=None, transport=self.sim.post)

    def test_unknown_field_is_rejected(self):
        self.client.enter()
        body = {"arena_id": "default", "robot_id": CFG["team_id"],
                "request_id": "x-1", "position": {"x": 0, "y": 0}, "channel": 1,
                "comment": "typo"}
        status, resp = self.sim.post("/measure", body)
        self.assertEqual(status, 200)
        self.assertFalse(resp["accepted"], "未知字段必须返回 accepted=false")

    def test_client_rejects_raises(self):
        self.client.enter()
        with self.assertRaises(RejectedError):
            # 伪造一个带未知字段的请求：直接走底层 _call
            payload = self.client._base("measure", "m-999")
            payload["position"] = {"x": 0, "y": 0}
            payload["channel"] = 1
            payload["oops"] = 1
            self.client._call("/measure", payload)

    def test_idempotent_retry_returns_first_response(self):
        self.client.enter()
        payload = self.client._base("measure", "m-fixed")
        payload["position"] = {"x": 100.0, "y": 0.0}
        payload["channel"] = 1
        first = self.sim.post("/measure", payload)
        again = self.sim.post("/measure", payload)
        self.assertEqual(first[1]["virtual_time_s"], again[1]["virtual_time_s"])
        self.assertAlmostEqual(self.sim.virtual_time_s, first[1]["virtual_time_s"], places=6)

    def test_request_id_reuse_with_different_content_is_409(self):
        self.client.enter()
        p1 = self.client._base("measure", "m-dup")
        p1["position"] = {"x": 0.0, "y": 0.0}
        p1["channel"] = 1
        self.sim.post("/measure", p1)
        p2 = dict(p1)
        p2["position"] = {"x": 500.0, "y": 0.0}
        status, _ = self.sim.post("/measure", p2)
        self.assertEqual(status, 409)

    def test_clear_does_not_switch_channel(self):
        self.client.enter()
        self.client.measure((0.0, 0.0), 5)
        self.assertEqual(self.sim.channel, 5)
        self.client.clear((0.0, 0.0), 9)
        self.assertEqual(self.sim.channel, 5, "/clear 不得改变测向机当前频道")


class TestCostModelMatchesSimulator(unittest.TestCase):
    """本地计时镜像必须与模拟器逐次一致（预测耗时 == 实际虚拟时间增量）。"""

    def test_sequence_matches(self):
        sim = shadow_mod.ShadowSim(CFG, mode="q3", seed=3, n_sources=12)
        cost = CostModel()
        client = SimClient(CFG, logger=None, transport=sim.post)
        client.enter_wait(1.0)
        cost.on_enter(0.0)
        seq = [("measure", (300.0, 400.0), 1), ("measure", (300.0, 400.0), 2),
               ("clear", (300.0, 0.0), 3), ("measure", (300.0, 0.0), 2)]
        for kind, pos, ch in seq:
            if kind == "measure":
                out = client.measure(pos, ch)
                pred = cost.predict_measure(pos, ch)
                cost.on_measure_accepted(pos, ch, out.virtual_time_s, pred["total_s"])
            else:
                out = client.clear(pos, ch)
                found = out.extra.get("result") == "success"
                pred = cost.predict_clear(pos, ch, found=found)
                cost.on_clear_accepted(pos, out.virtual_time_s, pred["total_s"])
        self.assertEqual(cost.mismatches, [], "计时镜像与模拟器不一致：%s" % cost.mismatches)
        self.assertAlmostEqual(cost.virtual_time_s, sim.virtual_time_s, places=6)
        self.assertEqual(cost.position, sim.position)
        self.assertEqual(cost.channel, sim.channel)


class TestEndToEnd(unittest.TestCase):
    def test_dryrun_clears_sources(self):
        sim = shadow_mod.ShadowSim(CFG, mode="q3", seed=11)
        runner, world, _ = build(CFG, sim.post, "dryrun", "q3", 11, None)
        stats = runner.run(enter_wait_s=1.0)
        gt = sim.ground_truth()
        truth = sum(1 for s in gt["sources"] if s["cleared"])
        self.assertEqual(stats["cleared_sources"], truth)
        self.assertGreaterEqual(truth / gt["n"], 0.8, "全向场景清除比例过低")
        self.assertEqual(stats["cost_model_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
