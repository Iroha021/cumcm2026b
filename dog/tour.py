"""巡游调度：把"去哪、测什么、清哪个"组织成一条滚动巡游。

设计要点（对应 TR 15-014 的 PlaceSensors + GatherData，但用于动态场景）：
  1) 节点只有三类：SWEEP（覆盖站点）、CLEAR（已定位待清除源）、VERIFY（收尾复核点）；
  2) **站点节点只建一次**（未访问才建）；到了站点要测哪些频道，是执行时才决定的
     "批量"，不预先建成独立节点 —— 这是消灭"专程三角定位"的关键；
  3) 站点与源**合并进同一条巡游**并用 TSP 排序，避免"先绕完站点再绕源"两趟；
  4) 巡游每步重规划（rolling horizon），新发现的信息立刻进模型。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import routing

Point = Tuple[float, float]


class Node(object):
    __slots__ = ("kind", "position", "channel", "value", "meta")

    def __init__(self, kind: str, position: Point, channel: Optional[int] = None,
                 value: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        self.kind = kind                  # 'sweep' | 'clear' | 'verify'
        self.position = (float(position[0]), float(position[1]))
        self.channel = channel
        self.value = float(value)
        self.meta = meta or {}

    def __repr__(self) -> str:
        return "<Node %s ch=%s pos=(%.0f,%.0f) v=%.2f>" % (
            self.kind, self.channel, self.position[0], self.position[1], self.value)

    def key(self) -> Tuple[str, int, int]:
        return (self.kind, int(round(self.position[0])), int(round(self.position[1])))


class TourPlanner(object):
    """滚动巡游：构建节点 → TSP 排序 → 返回下一个节点。"""

    def __init__(self, cfg: Dict[str, Any], world, cost, logger=None):
        self.cfg = cfg
        self.world = world
        self.cost = cost
        self.logger = logger
        pol = cfg["policy"]
        env = cfg["env"]
        self.arena_r = float(env["arena_radius_m"])
        self.r_min = float(env["recv_radius_min_m"])
        self.r_max = float(env["recv_radius_max_m"])
        self.cover_r = float(pol.get("cover_radius_m", 1000.0))
        self.min_angle = float(pol.get("refine_min_new_angle_deg", 20.0))
        self.v_sweep = float(pol.get("node_value_sweep", 1.0))
        self.v_clear = float(pol.get("node_value_clear", 3.0))
        self.v_verify = float(pol.get("node_value_verify", 0.3))
        self.min_probes_before_absent = int(pol.get("min_probes_before_absent", 8))
        self.cover_value_floor = float(pol.get("cover_value_floor", 1e-3))
        self.cover_pd_floor = float(pol.get("cover_pd_floor", 0.02))
        if bool(pol.get("use_greedy_cover", False)):
            p1 = routing.greedy_cover(self.arena_r, self.cover_r)
            p2 = []
        else:
            p1, p2 = routing.covering_stations(
                self.arena_r, self.cover_r, stagger=bool(pol.get("sweep_stagger", True)))
        self.stations_pass1, self.stations_pass2 = p1, p2
        self.stagger_max = int(pol.get("stagger_stations_max", 3))
        self.visited: set = set()
        self.phase = "sweep"
        self.verify_used = 0                    # 收尾复核已用节点数

    # ------------------------------------------------------------- 站点侧
    @staticmethod
    def _skey(p: Point) -> Tuple[int, int]:
        return (int(round(p[0])), int(round(p[1])))

    def _open_pass2(self) -> List[Point]:
        """pass1 走完后，按"还能服务多少未解决频道"挑前 K 个错位站点展开。

        这样在"首批覆盖站点就把事情办完"的局面下可以整趟省掉错位站点。
        """
        rest = [p for p in self.stations_pass2 if self._skey(p) not in self.visited]
        if not rest:
            return []
        scored = []
        for p in rest:
            v, _ = self._station_value(p)
            scored.append((v, p))
        scored.sort(key=lambda t: -t[0])
        return [p for v, p in scored[:self.stagger_max] if v > 1e-4]

    def unvisited_stations(self) -> List[Point]:
        rest1 = [p for p in self.stations_pass1 if self._skey(p) not in self.visited]
        if rest1:
            return rest1
        return self._open_pass2()

    @property
    def stations(self) -> List[Point]:
        """当前"计划内"的站点（pass1 + 已展开的 pass2），仅供日志/诊断。"""
        return list(self.stations_pass1) + list(self.stations_pass2)

    def mark_visited(self, position: Point) -> None:
        self.visited.add(self._skey(position))

    def nearest_unvisited_station(self) -> Optional[Point]:
        cur = self.cost.position
        rest = self.unvisited_stations()
        if not rest:
            return None
        return min(rest, key=lambda p: math.hypot(p[0] - cur[0], p[1] - cur[1]))

    # ------------------------------------------------------------- 节点构建
    def build_nodes(self, clear_candidates: Sequence[Tuple[int, Point, float]],
                    extra_stations: Optional[Sequence[Point]] = None) -> List[Node]:
        """clear_candidates: [(channel, 清除点, 成功概率)]
        extra_stations:  额外补充站点（巡游后期仍缺几何时用的专程点，如问题二点）
        """
        nodes: List[Node] = []
        for ch, pt, p_succ in clear_candidates:
            nodes.append(Node("clear", pt, ch, value=self.v_clear * p_succ,
                              meta={"p_success": p_succ}))
        # 覆盖站点：价值 = 在该站还能测到多少"未解决"东西（空则不再建）
        for p in self.unvisited_stations():
            v, info = self._station_value(p)
            if v > 1e-4:
                nodes.append(Node("sweep", p, value=self.v_sweep * v, meta=info))
        for p in (extra_stations or []):
            v, info = self._station_value(p)
            if v > 1e-4:
                nodes.append(Node("sweep", p, value=self.v_sweep * v,
                                  meta=dict(info, dedicated=True)))
        # 收尾复核
        if not nodes and self.phase == "sweep":
            self.phase = "verify"
        if self.phase == "verify":
            nodes.extend(self._verify_nodes())
        return nodes

    def _station_value(self, p: Point) -> Tuple[float, Dict[str, Any]]:
        """站点价值 = 该站最值得测的前 batch_max 个频道的价值之和。"""
        batch = self.batch_channels_at(p)
        if not batch:
            return 0.0, {"batch": 0}
        return float(sum(v for v, _ in batch)), {"batch": len(batch)}

    # ------------------------------------------------------- 站点批量（核心）
    def batch_channels_at(self, p: Point, keep_top: Optional[int] = None):
        """到站后该测哪些频道，按价值降序。

        优先级：
          1) 只有 1 条示向度、且在该站交会角 >= min_angle 的频道（顺路拿第二条）
          2) 从未测到信号的频道（覆盖式发现）
          3) >=2 条示向度但 GDOP 仍不达标的频道（顺带收紧）
        """
        w = self.world
        out: List[Tuple[float, int, str]] = []
        r_min, r_max = self.r_min, self.r_max
        u_star = float(self.cfg["policy"].get("u_star", 6.6e5))
        max_meas = int(self.cfg["policy"].get("max_measures_per_channel", 20))
        for ch in w.pending():
            b = w.beliefs[ch]
            if b.n_probes >= max_meas:
                continue
            nb = len(b.bearings)
            est = b.point_estimate() if nb > 0 else None
            d_est = math.hypot(p[0] - est[0], p[1] - est[1]) if est else None
            p_geo = (min(1.0, max(0.0, (r_max - d_est) / (r_max - r_min)))
                     if d_est is not None else None)
            if nb == 0:
                # 用解析网格判断"该点能不能测到"
                pd = float(b.detect_prob([[p[0], p[1]]])[0])
                v = float(self.cfg["policy"].get("value_discover", 0.3)) * b.p_exist * pd
                # 覆盖未完成的频道必须测：p_exist/pd 都会随无信号证据衰减，但
                # "判无源"之前必须在覆盖级站点集合上都测过，否则远端的源会被误判掉。
                # 故此处**不设后验门槛**，只要还没测够次数就排进本站批量。
                need_cov = b.n_probes < self.min_probes_before_absent
                if need_cov:
                    out.append((max(v, self.cover_value_floor), ch, "discover"))
                elif v > 1e-4:
                    out.append((v, ch, "discover"))
            elif p_geo and p_geo > 0:
                obs = [q for q, _ in b.bearings]
                if nb == 1:
                    ang = routing.crossing_angle_deg(obs[0], p, est)
                    if ang >= self.min_angle:
                        out.append((float(self.cfg["policy"].get("value_second_point", 1.0))
                                    * p_geo, ch, "triangulate"))
                else:
                    u_before = routing.best_pair_gdop(obs, est)
                    u_after = min(u_before, min(routing.gdop(o, p, est) for o in obs))
                    if u_after < u_before * 0.985:
                        gain = (u_before - u_after) / max(u_before, 1.0)
                        out.append((float(self.cfg["policy"].get("value_refine", 0.8))
                                    * p_geo * (0.4 + 3.0 * gain), ch, "refine"))
        out.sort(key=lambda t: -t[0])
        cap = int(self.cfg["policy"].get("batch_max", 20))
        if keep_top is not None:
            cap = min(cap, int(keep_top))
        return [(v, ch) for v, ch, _ in out[:cap]]

    # -------------------------------------------------------------- 复核
    def _verify_nodes(self) -> List[Node]:
        """收尾复核：对"被判无源"的频道，去最有可能漏掉它的位置补测一轮。"""
        if self.verify_used >= int(self.cfg["policy"].get("verify_max_nodes", 3)):
            return []
        w = self.world
        thr = float(self.cfg["policy"].get("absent_mass_threshold", 0.02))
        cands = [ch for ch in w.channels if ch not in w.cleared
                 and not w.beliefs[ch].bearings and w.beliefs[ch].p_exist <= thr]
        if not cands:
            return []
        cur = self.cost.position
        best = None
        for ch in cands:
            b = w.beliefs[ch]
            pts = [p for p in self.stations if self._skey(p) not in self.visited]
            pts = pts or list(self.stations)
            arr = [[p[0], p[1]] for p in pts]
            probs = b.detect_prob(arr)
            for i, p in enumerate(pts):
                v = float(b.p_exist) * float(probs[i])
                if v > 1e-6:
                    d = math.hypot(p[0] - cur[0], p[1] - cur[1])
                    if best is None or v / (d + 100.0) > best[0]:
                        best = (v / (d + 100.0), ch, p)
        if best is None:
            return []
        _, ch, p = best
        self.verify_used += 1
        return [Node("verify", p, ch, value=self.v_verify)]

    # ------------------------------------------------------------- 排序
    def order(self, nodes: Sequence[Node]) -> List[Node]:
        """按最近邻 + 2-opt 排序（每步从当前位置重新规划）。

        实测比较过"Or-opt"与"持久巡游 + 最廉插入 + 定期重优化"两种强化方案：
        前者只省 0.5% 里程却掉 0.8pp 清除比例；后者与逐步重规划无差别。
        故维持最简形式。
        """
        if len(nodes) <= 1:
            return list(nodes)
        pts = [n.position for n in nodes]
        order, _ = routing.tour_order(pts, self.cost.position)
        return [nodes[i] for i in order]
