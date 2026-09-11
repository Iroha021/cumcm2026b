"""2026 高教社杯 B 题 机器狗自动定位清除系统 —— 运行时包。

模块划分（见 docs 中的框架约定）：
    geom.py        问题一几何内核：半平面交、定位区域、直径、覆盖判定、域约束
    cost.py        虚拟时间与测向机频道/位置状态镜像（对齐附件 1 表 2）
    sim_client.py  唯一发 HTTP 的地方：4 条指令、字段白名单、幂等重试、双检查
    world.py       世界模型：每频道粒子信念 + 证据更新 + 存在性判定
    policy.py      决策层：选点/分区/巡游/停止条件（问题二打分函数在此）
    runner.py      串行主循环 + 预算管理 + 生命周期
    shadow.py      影子模拟器（物理 + 计时），作为可替换 transport 供 dryrun 使用
    log.py         JSONL 行为日志与运行报告
"""

__all__ = ["geom", "cost", "sim_client", "world", "policy", "runner", "shadow", "log"]
