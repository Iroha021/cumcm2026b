"""唯一入口。

    python run.py live    --case-code 1234        # 对接官方模拟器（先登录、开测试、等接口就绪）
    python run.py dryrun  --seed 1 --scenario q3  # 接自建影子模拟器，离线跑通全流程
    python run.py bench   --cases 40              # 影子模拟器批量跑，输出统计（调参用）
    python run.py q2                              # 问题二：候选区域计算与出图
    python run.py q1                              # 问题一：区域/直径/覆盖判定演示
    python run.py selftest                        # 离线自检：计时模型 / 几何 / 协议
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dog import geom, policy as policy_mod, shadow as shadow_mod, world as world_mod       # noqa: E402
from dog.cost import CostModel                                                             # noqa: E402
from dog.log import RunLogger                                                              # noqa: E402
from dog.runner import Runner                                                              # noqa: E402
from dog.sim_client import SimClient                                                       # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str = None) -> Dict[str, Any]:
    path = path or os.path.join(ROOT, "config.json")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_root"] = ROOT
    # 队号不写进仓库：优先取环境变量 CUMCM_TEAM_ID（config.json 里是 YOUR_TEAM_ID 占位符）
    env_id = os.environ.get("CUMCM_TEAM_ID")
    if env_id:
        cfg["team_id"] = env_id.strip()
    return cfg


# --------------------------------------------------------------------- 组装
def build(cfg: Dict[str, Any], transport, mode: str, scenario: str, seed: int, logger,
          session_tag: str = ""):
    world = world_mod.World(cfg, mode=scenario, seed=seed)
    cost = CostModel(speed_mps=cfg["env"]["speed_mps"],
                     detect_s=cfg["env"]["detect_s"],
                     switch_s=cfg["env"]["switch_s"],
                     optical_s=cfg["env"]["optical_s"],
                     clear_s=cfg["env"]["clear_s"],
                     start_pos=(0.0, 0.0),
                     start_channel=cfg["env"]["channel_min"])
    client = SimClient(cfg, logger=logger, transport=transport, session_tag=session_tag)
    pol = policy_mod.Policy(cfg, world, cost, logger=logger, mode=scenario, seed=seed)
    runner = Runner(cfg, client, world, cost, pol, logger, mode=mode)
    if logger is not None:
        # 参数快照：正式测试只有 3 次机会，事后必须能自证"这局跑的是哪个模式、哪些参数"
        logger.event("config_snapshot", scenario=scenario, run_mode=mode, seed=seed,
                     session_tag=session_tag, team_id=cfg["team_id"],
                     base_url=cfg["base_url"],
                     cover_radius_m=cfg["policy"].get("cover_radius_m"),
                     use_greedy_cover=cfg["policy"].get("use_greedy_cover"),
                     stagger_stations_max=cfg["policy"].get("stagger_stations_max"),
                     directional_model=(scenario == "q4"))
    return runner, world, client


def _new_session_tag(i: int) -> str:
    return "%04x" % (int(time.time() * 1000) & 0xFFFF ^ (i * 7919) & 0xFFFF)


# --------------------------------------------------------------------- 模式
def cmd_live(cfg, args):
    """对接官方模拟器。--sessions N 可连续跑 N 局：

    程序会自己等接口出现（连接被拒 = 测试还没开始，属正常），跑完一局落盘后
    继续等下一局。你只需要在模拟器界面上连点"开始测试"。
    """
    sessions = max(1, int(args.sessions))
    for i in range(sessions):
        tag = "live-%s%s" % (args.scenario, ("-r%d" % (i + 1)) if sessions > 1 else "")
        logger = RunLogger(os.path.join(cfg["_root"], cfg["logging"]["runs_dir"]),
                           tag=tag, case_code=args.case_code,
                           echo=bool(cfg["logging"].get("echo", True)))
        runner, _, _ = build(cfg, None, "live", args.scenario, args.seed, logger,
                             session_tag=_new_session_tag(i))
        if sessions > 1:
            print("=== 第 %d/%d 局：等待接口开放（请在模拟器界面点『开始』）===" % (i + 1, sessions),
                  flush=True)
        stats = runner.run(enter_wait_s=args.enter_wait)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_dryrun(cfg, args):
    sessions = max(1, int(args.sessions))
    for i in range(sessions):
        seed = args.seed + i
        sim = shadow_mod.ShadowSim(cfg, mode=args.scenario, seed=seed, n_sources=args.sources)
        logger = RunLogger(os.path.join(cfg["_root"], cfg["logging"]["runs_dir"]),
                           tag="dryrun-%s" % args.scenario,
                           echo=bool(cfg["logging"].get("echo", True)))
        runner, world, _ = build(cfg, sim.post, "dryrun", args.scenario, seed, logger)
        stats = runner.run(enter_wait_s=1.0)
        gt = sim.ground_truth()
        truth = sum(1 for s in gt["sources"] if s["cleared"])
        print(json.dumps({"seed": seed, "stats": stats,
                          "truth": {"n": gt["n"], "directional": gt["directional"],
                                    "cleared_truth": truth}}, ensure_ascii=False, indent=2))
    return 0


def cmd_bench(cfg, args):
    rows: List[Dict[str, Any]] = []
    for i in range(args.cases):
        seed = args.seed + i
        sim = shadow_mod.ShadowSim(cfg, mode=args.scenario, seed=seed,
                                   n_sources=args.sources)
        runner, world, _ = build(cfg, sim.post, "dryrun", args.scenario, seed, None)
        stats = runner.run(enter_wait_s=1.0)
        gt = sim.ground_truth()
        truth = sum(1 for s in gt["sources"] if s["cleared"])
        stats["truth_n"] = gt["n"]
        stats["truth_cleared"] = truth
        stats["recall"] = (truth / gt["n"]) if gt["n"] else 0.0
        stats["seed"] = seed
        rows.append(stats)
        print("[case %3d] seed=%d n=%d cleared=%d recall=%.3f vt=%.0f real=%.2fs reason=%s"
              % (i + 1, seed, gt["n"], truth, stats["recall"], stats["virtual_time_s"],
                 stats["program_real_time_s"], stats["ended_reason"]), flush=True)
    if rows:
        n = len(rows)
        print("\n=== 汇总（%d 局） ===" % n)
        print("平均清除比例: %.4f" % (sum(r["recall"] for r in rows) / n))
        print("平均虚拟时间: %.1f s" % (sum(r["virtual_time_s"] for r in rows) / n))
        print("平均平均定位清除时间: %.1f s" % (
            sum((r["avg_clear_virtual_s"] or 0) for r in rows) / n))
        print("平均动作数: %.1f" % (sum(r["actions"] for r in rows) / n))
        print("平均现实耗时: %.3f s" % (sum(r["program_real_time_s"] for r in rows) / n))
        print("全清局数: %d / %d" % (sum(1 for r in rows if r["recall"] >= 0.999), n))
        out = os.path.join(cfg["_root"], "runs", "bench-%d.json" % int(time.time()))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("明细已写入: %s" % out)
    return 0


def cmd_q2(cfg, args):
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import q2_candidate
    q2_candidate.main(cfg)
    return 0


def cmd_q1(cfg, args):
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import q1_region
    q1_region.main(cfg)
    return 0


def cmd_selftest(cfg, args):
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(ROOT, "tests"))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="2026 CUMCM B 题 机器狗程序")
    ap.add_argument("command", choices=["live", "dryrun", "bench", "q1", "q2", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenario", default="q3", choices=["q3", "q4"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--sources", type=int, default=None)
    ap.add_argument("--case-code", default="")
    ap.add_argument("--enter-wait", type=float, default=240.0)
    ap.add_argument("--sessions", type=int, default=1,
                    help="live/dryrun 连续跑几局（live 下每局需在界面点开始）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    _tid = cfg.get("team_id") or ""
    if args.command == "live" and not (_tid.isdigit() and len(_tid) == 12):
        print("队号未设置或不是 12 位数字：config.json 里是占位符 %r。" % _tid)
        print('请先执行  $env:CUMCM_TEAM_ID="<12 位队号>"  再运行 live（也可直接改 config.json）。')
        return 2
    return {"live": cmd_live, "dryrun": cmd_dryrun, "bench": cmd_bench,
            "q1": cmd_q1, "q2": cmd_q2, "selftest": cmd_selftest}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
