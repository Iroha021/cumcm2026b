"""计时模型自检：必须逐项对齐附件 1 表 2 的官方示例。"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog.cost import CostModel  # noqa: E402


class TestCostModel(unittest.TestCase):
    """附件 1 表 2 / 附件 2 第 10 节的五步示例：
    /enter → measure(300,400) ch1 → measure(300,400) ch2 →
    clear(300,0) ch3(未发现) → measure(300,0) ch2 → /exit
    期望虚拟时刻：105 / 111 / 194 / 199
    """

    def test_official_example(self):
        cm = CostModel()
        # 步骤 2：从 (0,0) 到 (300,400)，500 m，ch1 = 初始频道
        p = cm.predict_measure((300.0, 400.0), 1)
        self.assertAlmostEqual(p["move_s"], 100.0, places=6)
        self.assertAlmostEqual(p["switch_s"], 0.0, places=6)
        self.assertAlmostEqual(p["total_s"], 105.0, places=6)
        cm.on_measure_accepted((300.0, 400.0), 1, 105.0, p["total_s"])
        self.assertAlmostEqual(cm.virtual_time_s, 105.0, places=6)

        # 步骤 3：位置不变，换到 ch2 ⇒ 1 s 切换
        p = cm.predict_measure((300.0, 400.0), 2)
        self.assertAlmostEqual(p["total_s"], 6.0, places=6)
        cm.on_measure_accepted((300.0, 400.0), 2, 111.0, p["total_s"])

        # 步骤 4：clear 到 (300,0)，400 m ⇒ 80 s 移动 + 3 s 精确定位（未发现）
        p = cm.predict_clear((300.0, 0.0), 3, found=False)
        self.assertAlmostEqual(p["move_s"], 80.0, places=6)
        self.assertAlmostEqual(p["total_s"], 83.0, places=6)
        cm.on_clear_accepted((300.0, 0.0), 194.0, p["total_s"])
        self.assertEqual(cm.channel, 2, "clear 不得改变当前频道")

        # 步骤 5：原地测 ch2 ⇒ 0 移动 0 切换 5 s
        p = cm.predict_measure((300.0, 0.0), 2)
        self.assertAlmostEqual(p["total_s"], 5.0, places=6)
        cm.on_measure_accepted((300.0, 0.0), 2, 199.0, p["total_s"])

        self.assertAlmostEqual(cm.virtual_time_s, 199.0, places=6)
        self.assertEqual(len(cm.mismatches), 0)

    def test_clear_does_not_change_channel(self):
        cm = CostModel()
        cm.on_measure_accepted((0.0, 0.0), 7, 5.0, 5.0)
        cm.on_clear_accepted((0.0, 0.0), 8.0, 3.0)
        self.assertEqual(cm.channel, 7)
        p = cm.predict_measure((0.0, 0.0), 7)
        self.assertAlmostEqual(p["switch_s"], 0.0)

    def test_mismatch_detected(self):
        cm = CostModel()
        p = cm.predict_measure((300.0, 400.0), 1)
        cm.on_measure_accepted((300.0, 400.0), 1, 999.0, p["total_s"])
        self.assertEqual(len(cm.mismatches), 1)


if __name__ == "__main__":
    unittest.main()
