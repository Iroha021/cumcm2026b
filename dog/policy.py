"""决策层：问题二选点打分 + 问题三/四的滚动策略。

对外只有两个入口：
    select_second_point(cfg, P1, th1)  —— 问题二的完整算法（离线与在线共用）
    Policy.decide()                    —— 每次返回一个 Action（纯函数式，便于回放）

Q2 目标：在保证第二点能测到的前提下，使交会区域直径的期望最小。
    D(φ=90°) = 2√2·δ·r （已数值验证），故"正交"最优；
    但 r 未知且探测半径 R∈[1000,1500] 有限，垂直外推会拉大 |S-P2| 导致测不到，
    最优解被拉回到 β ≈ 45°，d ≈ 先验均值 r̂ 的 1.1~1.3 倍。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geom, routing, tour

Point = Tuple[float, float]

PENALTY_D = 3000.0


# --------------------------------------------------------------------- 批量几何
def batch_region_diameter(P1: Point, th1_deg: float, P2: Point,
                          th2_deg: np.ndarray, delta_deg: float,
                          penalty: float = PENALTY_D) -> np.ndarray:
    """两站定位区域直径的向量化计算（一次算 N 个样本）。

    与 geom.region_from_bearings 同源：四角 = 两条边界线的 4 个交点，
    保留同时落在两扇形内的点，直径 = 这些点两两距离的最大值。
    """
    p1 = np.asarray(P1, dtype=float)
    p2 = np.asarray(P2, dtype=float)
    th2 = np.asarray(th2_deg, dtype=float)
    n = th2.shape[0]

    def dvec(deg):
        r = np.radians(deg)
        return np.stack([np.cos(r), np.sin(r)], axis=-1)

    d1m = dvec(th1_deg - delta_deg)
    d1p = dvec(th1_deg + delta_deg)
    d2m = dvec(th2 - delta_deg)
    d2p = dvec(th2 + delta_deg)

    cands: List[Tuple[np.ndarray, np.ndarray]] = []
    for d1 in (d1m, d1p):
        for d2 in (d2m, d2p):
            den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
            ok = np.abs(den) > 1e-12
            w = p2 - p1
            safe = np.where(ok, den, 1.0)
            t = (w[0] * d2[:, 1] - w[1] * d2[:, 0]) / safe
            x = p1[None, :] + t[:, None] * d1[None, :]
            v1 = x - p1[None, :]
            in1 = ((d1m[0] * v1[:, 1] - d1m[1] * v1[:, 0] >= -1e-9) &
                   (d1p[0] * v1[:, 1] - d1p[1] * v1[:, 0] <= 1e-9))
            v2 = x - p2[None, :]
            in2 = ((d2m[:, 0] * v2[:, 1] - d2m[:, 1] * v2[:, 0] >= -1e-9) &
                   (d2p[:, 0] * v2[:, 1] - d2p[:, 1] * v2[:, 0] <= 1e-9))
            cands.append((x, ok & in1 & in2))

    # 定位区域的顶点还可能落在某一检测点自身（当该点落在另一扇形内时，
    # 区域退化为三角形，如源离该站极近的情形），必须一并纳入候选。
    apex1 = np.broadcast_to(p1, (n, 2))
    v12 = p1[None, :] - p2
    in1_at_p2 = ((d2m[:, 0] * v12[:, 1] - d2m[:, 1] * v12[:, 0] >= -1e-9) &
                 (d2p[:, 0] * v12[:, 1] - d2p[:, 1] * v12[:, 0] <= 1e-9))
    cands.append((apex1, in1_at_p2))
    apex2 = np.broadcast_to(p2, (n, 2))
    v21 = apex2 - p1[None, :]
    in2_at_p1 = ((d1m[0] * v21[:, 1] - d1m[1] * v21[:, 0] >= -1e-9) &
                 (d1p[0] * v21[:, 1] - d1p[1] * v21[:, 0] <= 1e-9))
    cands.append((apex2, in2_at_p1))

    best = np.zeros(n)
    nvalid = np.zeros(n, dtype=int)
    for x, v in cands:
        nvalid += v.astype(int)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            xi, vi = cands[i]
            xj, vj = cands[j]
            dist = np.hypot(xi[:, 0] - xj[:, 0], xi[:, 1] - xj[:, 1])
            mask = vi & vj
            best = np.where(mask & (dist > best), dist, best)
    # 无界判据（两站，方向区间有公共方向 ⇔ 圆形角距 <= 2δ）：此时直径发散
    dd = np.abs(((th2 - th1_deg) + 180.0) % 360.0 - 180.0)
    unbounded = dd <= (2.0 * delta_deg + 1e-9)
    ok = (nvalid >= 3) & (~unbounded)
    return np.where(ok, best, penalty)


# ------------------------------------------------------------------ 问题二算法
def sample_detection_conditioned(cfg: Dict[str, Any], P1: Point, th1_deg: float,
                                 n: int, rng: np.random.Generator,
                                 directional: bool = False,
                                 p_omni: float = 0.5) -> Dict[str, np.ndarray]:
    """按"在 P1 已测到信号"的条件采样真实源。

    directional=True 时同时采样类型与定向方向 α：既然在 P1 测到了，
    α 必落在"以 P1 为中心、张角 180°"的那个弧内（覆盖条件是半平面）。
    """
    env = cfg["env"]
    delta = float(env["bearing_error_deg"])
    r_min = float(env["recv_radius_min_m"])
    r_max = float(env["recv_radius_max_m"])
    arena_r = float(env["arena_radius_m"])
    R = rng.uniform(r_min, r_max, n)
    th_true = th1_deg + rng.uniform(-delta, delta, n)
    chord = np.array([geom.chord_length(P1, t, (0.0, 0.0), arena_r) for t in th_true])
    rmax = np.maximum(np.minimum(R, chord), 1.0)
    r = rmax * np.sqrt(rng.random(n))
    rad = np.radians(th_true)
    S = np.stack([P1[0] + r * np.cos(rad), P1[1] + r * np.sin(rad)], axis=1)
    eps2 = rng.uniform(-delta, delta, n)
    out = {"R": R, "r": r, "S": S, "eps2": eps2}
    if directional:
        omni = rng.random(n) < p_omni
        # P1 相对源的方向角，α 需落在该角 ±90°（含边界）
        to_p1 = np.degrees(np.arctan2(P1[1] - S[:, 1], P1[0] - S[:, 0]))
        alpha = np.radians(to_p1 + rng.uniform(-90.0, 90.0, n))
        out["omni"] = omni
        out["alpha"] = alpha
    return out


def select_second_point(cfg: Dict[str, Any], P1: Point, th1_deg: float,
                        rng: Optional[np.random.Generator] = None,
                        n_mc: Optional[int] = None,
                        beta_step_deg: Optional[float] = None,
                        d_step_m: Optional[float] = None,
                        refine: bool = True,
                        directional: bool = False) -> Dict[str, Any]:
    """问题二：选择第二个检测点，并给出候选区域。

    directional=True（问题四）时额外考虑定向覆盖：第二点必须同时位于源的
    覆盖半平面内才可能测到，这会显著改变候选区域（偏向"从可见侧"布置）。
    """
    q2 = cfg["q2"]
    env = cfg["env"]
    delta = float(env["bearing_error_deg"])
    rng = rng or np.random.default_rng(20260910)
    n_mc = int(n_mc or q2["n_mc"])
    beta_step = float(beta_step_deg or q2["beta_step_deg"])
    d_step = float(d_step_m or q2["d_step_m"])
    d_min = float(q2["d_min_m"])
    d_max = float(q2["d_max_m"])
    eta = float(q2["eta"])
    pi_min = float(q2["pi_min"])

    smp = sample_detection_conditioned(cfg, P1, th1_deg, n_mc, rng,
                                       directional=directional)
    R, S, eps2 = smp["R"], smp["S"], smp["eps2"]

    def evaluate(d: float, beta_deg: float):
        b = math.radians(th1_deg + beta_deg)
        P2 = (P1[0] + d * math.cos(b), P1[1] + d * math.sin(b))
        rel = S - np.array(P2)
        dist = np.hypot(rel[:, 0], rel[:, 1])
        det = dist <= R
        if directional:
            al = smp["alpha"]
            proj = (-rel[:, 0]) * np.cos(al) + (-rel[:, 1]) * np.sin(al)
            det = det & (smp["omni"] | (proj >= 0.0))
        th2 = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) + eps2
        D = batch_region_diameter(P1, th1_deg, P2, th2, delta)
        good = det & (D < PENALTY_D)
        pi = float(det.mean())
        if good.sum() == 0:
            return pi, None, None, None
        # 交会角（源处两条射线的夹角）
        a1 = np.degrees(np.arctan2(S[:, 1] - P1[1], S[:, 0] - P1[0]))
        a2 = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        phi = np.abs((a2 - a1 + 180.0) % 360.0 - 180.0)[good]
        return pi, float(D[good].mean()), float(np.percentile(D[good], 90)), float(phi.mean())

    def sweep(betas: Sequence[float], ds: Sequence[float]):
        rows = []
        for beta in betas:
            for d in ds:
                pi, j, p90, phi = evaluate(d, beta)
                if j is None:
                    continue
                rows.append({"d": d, "beta": beta, "pi": pi, "E_D": j,
                             "P90_D": p90, "phi": phi})
        return rows

    betas = np.arange(0.0, 180.0, beta_step)
    ds = np.arange(max(50.0, d_min), d_max + 1e-9, d_step)
    rows = sweep(betas, ds)
    feas = [r for r in rows if r["pi"] >= pi_min and r["phi"] >= float(q2["phi_min_deg"])]
    pool = feas if feas else rows
    pool.sort(key=lambda r: (r["E_D"], r["d"]))
    best = pool[0]

    if refine:
        b0, d0 = best["beta"], best["d"]
        fine_b = np.arange(max(0.0, b0 - beta_step), min(180.0, b0 + beta_step) + 1e-9, beta_step / 2.0)
        fine_d = np.arange(max(50.0, d0 - d_step), d0 + d_step + 1e-9, d_step / 2.0)
        rows2 = sweep(fine_b, fine_d)
        feas2 = [r for r in rows2 if r["pi"] >= pi_min and r["phi"] >= float(q2["phi_min_deg"])]
        pool2 = (feas2 if feas2 else rows2) + rows
        pool2.sort(key=lambda r: (r["E_D"], r["d"]))
        best = pool2[0]
        rows = rows + rows2

    jmin = best["E_D"]
    region = [r for r in rows
              if r["E_D"] <= (1.0 + eta) * jmin and r["pi"] >= pi_min]
    betas_ok = [r["beta"] for r in region]
    ds_ok = [r["d"] for r in region]
    b = math.radians(th1_deg + best["beta"])
    P2 = (P1[0] + best["d"] * math.cos(b), P1[1] + best["d"] * math.sin(b))
    return {
        "P2": P2, "d": best["d"], "beta": best["beta"],
        "E_D": best["E_D"], "P90_D": best["P90_D"], "pi": best["pi"], "phi": best["phi"],
        "candidate_region": {
            "beta_min": min(betas_ok) if betas_ok else None,
            "beta_max": max(betas_ok) if betas_ok else None,
            "d_min": min(ds_ok) if ds_ok else None,
            "d_max": max(ds_ok) if ds_ok else None,
        },
        "grid": rows,
        "prior_r_mean": float(smp["r"].mean()),
    }


# --------------------------------------------------------------------- 动作
class Action(object):
    __slots__ = ("kind", "position", "channel", "reason", "expected_cost_s", "meta")

    def __init__(self, kind: str, position: Optional[Point] = None, channel: Optional[int] = None,
                 reason: str = "", expected_cost_s: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        self.kind = kind                  # 'measure' | 'clear' | 'exit'
        self.position = position
        self.channel = channel
        self.reason = reason
        self.expected_cost_s = expected_cost_s
        self.meta = meta or {}

    def __repr__(self) -> str:
        return "<Action %s ch=%s pos=%s %s cost=%.1f>" % (
            self.kind, self.channel,
            None if self.position is None else (round(self.position[0], 1), round(self.position[1], 1)),
            self.reason, self.expected_cost_s)


# --------------------------------------------------------------------- 策略
class Policy(object):
    def __init__(self, cfg: Dict[str, Any], world, cost, logger=None, mode: str = "q3",
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.world = world
        self.cost = cost
        self.logger = logger
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.pol = cfg["policy"]
        self.env = cfg["env"]
        self.arena_r = float(self.env["arena_radius_m"])
        self.clear_ready_d = float(self.pol["clear_ready_region_diameter_m"])
        self.absent_thr = float(self.pol["absent_mass_threshold"])
        self.measure_cap = int(cfg["budget"].get("max_consecutive_measures_per_stop", 20))
        self.clear_radius = float(self.env["clear_radius_m"])
        self.planner = tour.TourPlanner(cfg, world, cost, logger=logger)
        self._cur_node = None
        self._cur_batch: List[int] = []
        self._cur_reason = ""
        self.measure_streak = 0
        self.last_position: Point = (0.0, 0.0)
        self.q2_cache: Dict[Any, Dict[str, Any]] = {}
        self.clear_queue: Dict[int, List[Point]] = {}
        self.clear_attempts: Dict[int, int] = {}
        self.clear_tried_bearings: Dict[int, int] = {}
        self.probe_count: Dict[int, int] = {}
        self.last_miss_channel: Optional[int] = None   # 上一动作未命中的频道（就地补扫用）
        self.retry_streak = 0
        self.enable_clear_all = False

    # ------------------------------------------------------------ 候选探测点
    # （旧的固定探测栅格已由 tour.TourPlanner 的"覆盖站点"取代）

    # --------------------------------------------------------------- 主决策
    def decide(self) -> Action:
        """滚动巡游：
            ① 5 m 内直接清除（必然成功，最优先）
            ② 就地清除（清除点就在当前站点附近，零移动成本）
            ③ 完成当前站点的批量测量
            ④ 选下一个巡游节点（覆盖站点 / 已定位源 / 收尾复核点）
            ⑤ 无节点可做 → 退出
        """
        w, c = self.world, self.cost
        act = self._near_direct()
        if act:
            return act
        act = self._inline_retry()
        if act:
            return act
        act = self._inline_clear()
        if act:
            return act
        for _ in range(12):                      # 允许连续跳过若干"已无价值"的站点
            act = self._batch_action()
            if act:
                return act
            node = self._select_node()
            if node is None:
                return Action("exit", reason="no_action",
                              meta={"stats": w.stats(), "virtual_time_s": c.virtual_time_s})
            if node.kind == "clear":
                return self._clear_action_for(node)
            # sweep / verify：到站后开始批量；批量为空则跳过该站
            self._cur_node = node
            self._cur_batch = [ch for _, ch in
                               self.planner.batch_channels_at(node.position)]
            self._cur_reason = node.kind
            if not self._cur_batch:
                self.planner.mark_visited(node.position)
                self._cur_node = None
                continue
        return Action("exit", reason="plan_loop",
                      meta={"stats": w.stats(), "virtual_time_s": c.virtual_time_s})

    # ------------------------------------------------------------ 站点批量
    def _batch_action(self) -> Optional[Action]:
        if self._cur_node is None:
            return None
        pos = self._cur_node.position
        max_meas = int(self.pol.get("max_measures_per_channel", 20))
        while self._cur_batch:
            ch = self._cur_batch.pop(0)
            if ch in self.world.cleared:
                continue
            if self.world.beliefs[ch].n_probes >= max_meas:
                continue
            cost = self.cost.predict_measure(pos, ch)["total_s"]
            return Action("measure", pos, ch, reason=self._cur_reason or "batch",
                          expected_cost_s=cost,
                          meta={"station": (round(pos[0], 1), round(pos[1], 1)),
                                "batch_left": len(self._cur_batch)})
        self.planner.mark_visited(pos)
        self._cur_node = None
        self._cur_batch = []
        return None

    # ------------------------------------------------------------ 节点选择
    def _select_node(self):
        if self._cur_node is not None and self._cur_batch:
            return None                          # 当前站点还没测完
        extra = self._extra_stations()
        nodes = self.planner.build_nodes(self._clear_candidates(), extra_stations=extra)
        if not nodes:
            return None
        ordered = self.planner.order(nodes)
        return ordered[0] if ordered else None

    def _extra_stations(self) -> List[Point]:
        """巡游走完覆盖站点后仍不满足条件的频道 → 追加专程补测点：

          · 只有 1 条示向度：用问题二的第二检测点（交会角最优）；
          · >=2 条但几何太差（不确定度 > clear_ready 阈值）：用 GDOP+120° 补点。
        """
        out: List[Point] = []
        cap = int(self.pol.get("max_dedicated_stations", 6))
        r_min = float(self.env["recv_radius_min_m"])
        u_star = float(self.pol.get("u_star", 6.6e5))
        # 档 3A：先安排"必然存在、只缺第二条示向度"的频道（见 World.second_point_channels），
        # 它们每个节点都直接对应一个可清除的真实源；再安排常规的 pending 频道补测点。
        chans = self.world.second_point_channels(
            int(self.pol.get("max_second_point_nodes", 3)))
        for ch in self.world.pending():
            if ch not in chans:
                chans.append(ch)
        for ch in chans:
            if len(out) >= cap:
                break
            b = self.world.beliefs[ch]
            nb = len(b.bearings)
            if nb == 0:
                continue
            if nb >= 2 and b.uncertainty_m() <= self.clear_ready_d:
                continue                          # 够准了，只等清除
            if b.n_probes >= int(self.pol.get("max_measures_per_channel", 20)):
                continue
            if nb == 1:
                P1, th1 = b.bearings[0]
                key = (round(P1[0], 1), round(P1[1], 1), round(th1, 2))
                plan = self.q2_cache.get(key)
                if plan is None:
                    plan = select_second_point(self.cfg, P1, th1, rng=self.rng,
                                               directional=(self.mode == "q4"))
                    self.q2_cache[key] = plan
                    if self.logger:
                        self.logger.event("q2", channel=ch, P1=(round(P1[0], 2), round(P1[1], 2)),
                                          theta=round(th1, 3), second_point=plan["P2"],
                                          d=round(plan["d"], 1), beta=round(plan["beta"], 1),
                                          E_D=round(plan["E_D"], 2), pi=round(plan["pi"], 3),
                                          phi=round(plan["phi"], 1),
                                          candidate_region=plan["candidate_region"])
                out.append((float(plan["P2"][0]), float(plan["P2"][1])))
            else:
                est = b.point_estimate()
                if est is None:
                    continue
                p = routing.choose_refine_point(est, [q for q, _ in b.bearings], u_star, r_min)
                if p is not None:
                    out.append((float(p[0]), float(p[1])))
        return out

    # ------------------------------------------------------- 就地清除
    def _inline_retry(self) -> Optional[Action]:
        """清除未命中后就地补扫。

        未命中只排除掉 20 m 邻域，新的最佳清除点通常离当前位置只有几十米；
        但如果把它重新塞回巡游队列，就可能被 TSP 排到别的站点之后，
        一次本该 40 m 的补扫变成几百米转场——实测这正是局间差异（5.4 vs 10.9 km
        清除里程）的主要来源。因此未命中后优先就地连扫，超过上限才交回巡游。
        """
        if self.last_miss_channel is None:
            return None
        cap = int(self.pol.get("inline_retry_max", 3))
        if self.retry_streak >= cap:
            return None
        ch = self.last_miss_channel
        cur = self.cost.position
        for c, pt, p_succ in self._clear_candidates():
            if c != ch:
                continue
            d = math.hypot(pt[0] - cur[0], pt[1] - cur[1])
            if d > float(self.pol.get("inline_retry_radius_m", 150.0)):
                return None
            cost = self.cost.predict_clear(pt, ch, found=True)["total_s"]
            return Action("clear", pt, ch, reason="retry_inline", expected_cost_s=cost,
                          meta={"p_success": round(p_succ, 3), "hop_m": round(d, 1),
                                "retry": self.retry_streak + 1})
        return None

    def _inline_clear(self) -> Optional[Action]:
        """清除点就在脚边（<=60 m）时先清掉，避免为了它专门跑一趟。"""
        cur = self.cost.position
        for ch, pt, p_succ in self._clear_candidates():
            if math.hypot(pt[0] - cur[0], pt[1] - cur[1]) <= 60.0:
                return self._clear_action_for(
                    tour.Node("clear", pt, ch, value=0.0, meta={"p_success": p_succ}))
        return None


    # ------------------------------------------------------------ 各分支
    def _near_direct(self) -> Optional[Action]:
        for ch in self.world.near_hits():
            b = self.world.beliefs[ch]
            pos = [e["pos"] for e in b.evidence if e["kind"] == "near"][-1]
            cost = self.cost.predict_clear(pos, ch, found=True)
            return Action("clear", pos, ch, reason="near_direct",
                          expected_cost_s=cost["total_s"], meta={"guaranteed": True})
        return None

    def _clear_candidates(self):
        """返回 [(channel, 清除点, 一次成功概率)]，供巡游节点使用。

        清除点不取"区域质心"（区域直径 > 40 m 时质心不保证落在 20 m 内），而是
        直接最大化**后验成功概率**：取 20 m 半径内覆盖粒子数最多的位置。
        每次未命中都会以 |S-P| > 20 的形式收缩后验，于是同一位置不会被重复尝试。
        """
        p_min = float(self.pol.get("clear_min_success_prob", 0.35))
        max_attempts = int(self.pol.get("clear_max_attempts_per_channel", 6))
        out = []
        for ch in self.world.pending():
            b = self.world.beliefs[ch]
            if len(b.bearings) < 2:
                continue
            # 同一批示向度下最多尝试若干次清除（每次未命中都会收缩后验与排除点，
            # 因此重试是有信息的；但不能无限重复，超过上限就转去补几何）
            tried = self.clear_attempts.get(ch, 0)
            mark = self.clear_tried_bearings.get(ch, -1)
            if len(b.bearings) == mark and tried >= max_attempts:
                continue
            pt, p_succ = self._best_clear_point(ch)
            if pt is None or p_succ < p_min:
                continue
            out.append((ch, pt, p_succ))
        return out

    def _clear_action_for(self, node) -> Action:
        ch = node.channel
        cost = self.cost.predict_clear(node.position, ch, found=True)["total_s"]
        self.clear_tried_bearings[ch] = len(self.world.beliefs[ch].bearings)
        return Action("clear", node.position, ch, reason="localized", expected_cost_s=cost,
                      meta={"p_success": round(float(node.meta.get("p_success", 0.0)), 3),
                            "uncertainty_m": round(self.world.beliefs[ch].uncertainty_m(), 1)})

    def _best_clear_point(self, ch: int):
        """返回 (清除点, 该点一次成功的后验概率)。

        清除点 = "以该点为心、半径 20 m 的圆盘覆盖后验质量最多"的位置：
        粒子云携带真实的距离先验权重（∝ r），比"区域质心"更贴近真源分布；
        已尝试未命中的点周围不再作为候选，保证每次都换地方。
        """
        b = self.world.beliefs[ch]
        r = float(self.env["clear_radius_m"])
        # 区域直径 <= 20 m 时，区域内任意点都在真源 20 m 内 ⇒ 必然一次命中。
        # 这条是"可证明"的，优先于任何基于粒子云的启发式（粒子云在退化重采样后
        # 可能漂出可行域，历史上正是它导致了 D=7 m 却连续 3 次清除失败）。
        reg = b.region()
        if reg is not None and reg.valid and reg.diameter() <= r:
            c = reg.centroid()
            if c is not None:
                return (float(c[0]), float(c[1])), 1.0
        if b.src is None or b.src["pos"].shape[0] == 0:
            est = b.point_estimate()
            return (est, 0.5) if est else (None, 0.0)
        pos = b.src["pos"]
        lo = pos.min(axis=0) - r
        hi = pos.max(axis=0) + r
        ext = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        step = max(r, ext / 30.0)
        xs = np.arange(lo[0], hi[0] + 1e-9, step)
        ys = np.arange(lo[1], hi[1] + 1e-9, step)
        cand = [(float(x), float(y)) for x in xs for y in ys]
        # 已尝试未命中的点周围 0.95r 内不再作为候选（否则会在原地反复空扫）
        if b.clear_misses:
            keep = [p for p in cand
                    if all(math.hypot(p[0] - m[0], p[1] - m[1]) > 0.95 * r
                           for m in b.clear_misses)]
            if keep:
                cand = keep
        if not cand:
            est = b.point_estimate()
            return (est, 0.3) if est else (None, 0.0)
        arr = np.asarray(cand, dtype=float)
        d = np.hypot(arr[:, None, 0] - pos[None, :, 0], arr[:, None, 1] - pos[None, :, 1])
        cnt = (d <= r).sum(axis=1)
        if cnt.size == 0:
            return None, 0.0
        i = int(np.argmax(cnt))
        return (float(arr[i, 0]), float(arr[i, 1])), float(cnt[i]) / float(pos.shape[0])

    # ---------------------------------------------------------------- 反馈
    def q2_second_count(self, ch: int) -> int:
        return self.probe_count.get(ch, 0)

    def note(self, action: Action, outcome) -> None:
        if outcome is None:
            return
        if action.kind == "measure":
            if action.channel is not None:
                self.probe_count[action.channel] = self.probe_count.get(action.channel, 0) + 1
            same = (self.last_position is not None and action.position is not None and
                    math.hypot(action.position[0] - self.last_position[0],
                               action.position[1] - self.last_position[1]) < 1e-6)
            self.measure_streak = self.measure_streak + 1 if same else 1
            self.last_position = action.position or self.last_position
        else:
            self.measure_streak = 0
            self.last_position = action.position or self.last_position
            if action.kind == "clear" and action.channel is not None:
                self.clear_attempts[action.channel] = self.clear_attempts.get(action.channel, 0) + 1
                if getattr(outcome, "result", None) == "success":
                    self.clear_queue.pop(action.channel, None)
                    self.last_miss_channel = None
                    self.retry_streak = 0
                else:
                    # 未命中：记录频道与连扫次数，供"就地补扫"优先就地连扫
                    self.last_miss_channel = action.channel
                    self.retry_streak = (self.retry_streak + 1
                                         if action.reason == "retry_inline" else 1)
                    if self.logger:
                        self.logger.event("clear_miss", channel=action.channel,
                                          pos=action.position,
                                          attempts=self.clear_attempts[action.channel],
                                          retry=self.retry_streak)

    def should_stop(self) -> Optional[str]:
        w = self.world
        if not w.pending():
            return "all_resolved"
        if len(self.world.channels) - len(w.cleared) <= len(w.absent(self.absent_thr)):
            return "all_resolved"
        return None


def _grid_in_convex(poly: Sequence[Point], step: float, cap: int = 4000) -> List[Point]:
    """凸多边形内的栅格点（步长 step 米）。"""
    if len(poly) < 3 or step <= 0:
        return []
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if (x1 - x0) / step * (y1 - y0) / step > cap:
        step = math.sqrt(max(1.0, (x1 - x0) * (y1 - y0) / cap))
    out: List[Point] = []
    x = x0
    while x <= x1 + 1e-9:
        y = y0
        while y <= y1 + 1e-9:
            if _point_in_convex(poly, (x, y)):
                out.append((x, y))
            y += step
        x += step
    if not out:                                  # 区域太薄，栅格落空 → 用质心
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        out = [(cx, cy)]
    return out


def _point_in_convex(poly: Sequence[Point], p: Point) -> bool:
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-9:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True
