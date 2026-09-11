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

from . import geom

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
        self.measure_cap = int(cfg["budget"].get("max_consecutive_measures_per_stop", 6))
        self.clear_radius = float(self.env["clear_radius_m"])
        self.probe_grid = self._make_probe_grid()
        self.measure_streak = 0
        self.last_position: Point = (0.0, 0.0)
        self.q2_cache: Dict[Any, Dict[str, Any]] = {}
        self.clear_queue: Dict[int, List[Point]] = {}
        self.clear_attempts: Dict[int, int] = {}
        self.clear_tried_bearings: Dict[int, int] = {}
        self.probe_count: Dict[int, int] = {}
        self.enable_clear_all = False

    # ------------------------------------------------------------ 候选探测点
    def _make_probe_grid(self) -> List[Point]:
        """六边形栅格覆盖目标区：任意点都有候选探测点在其 ~s/√3 范围内。
        间距 s 默认 800 m：保证圆盘内任意源都落在某个候选点 1000 m 内（必然可测）。"""
        s = float(self.pol.get("probe_grid_spacing_m", 800.0))
        dy = s * math.sqrt(3.0) / 2.0
        pts: List[Point] = []
        ny = int(self.arena_r / dy) + 1
        nx = int(self.arena_r / s) + 1
        for iy in range(-ny, ny + 1):
            y = iy * dy
            off = (s / 2.0) if (iy % 2) else 0.0
            for ix in range(-nx, nx + 1):
                x = ix * s + off
                if math.hypot(x, y) <= self.arena_r:
                    pts.append((x, y))
        return pts

    # --------------------------------------------------------------- 主决策
    def decide(self) -> Action:
        w, c = self.world, self.cost
        # 1) 5 m 内直接清除（最便宜且必然成功）
        act = self._near_direct()
        if act:
            return act
        # 2) 已可精确定位 → 前去清除
        act = self._clear_action()
        if act:
            return act
        # 3) 信息型探测：在"探哪个频道 / 去哪探"上取单位时间收益最大者
        act = self._probe_action()
        if act:
            return act
        # 4) 无事可做
        return Action("exit", reason="no_action",
                      meta={"stats": w.stats(), "virtual_time_s": c.virtual_time_s})

    # ------------------------------------------------------------ 各分支
    def _near_direct(self) -> Optional[Action]:
        for ch in self.world.near_hits():
            b = self.world.beliefs[ch]
            pos = [e["pos"] for e in b.evidence if e["kind"] == "near"][-1]
            cost = self.cost.predict_clear(pos, ch, found=True)
            return Action("clear", pos, ch, reason="near_direct",
                          expected_cost_s=cost["total_s"], meta={"guaranteed": True})
        return None

    def _clear_action(self) -> Optional[Action]:
        """选择要清除的源与清除点。

        清除点不用"区域质心"（区域直径 > 40 m 时质心不保证落在 20 m 内），而是
        直接最大化**后验成功概率**：取 20 m 半径内覆盖粒子数最多的位置。
        每次未命中都会以 |S-P| > 20 的形式收缩后验，于是同一位置不会被重复尝试。
        """
        p_min = float(self.pol.get("clear_min_success_prob", 0.5))
        w = self.pol.get("clear_score_weight_s", 40.0)
        max_attempts = int(self.pol.get("clear_max_attempts_per_channel", 3))
        span_min = float(self.pol.get("min_bearing_span_deg", 15.0))
        best = None
        for ch in self.world.pending():
            b = self.world.beliefs[ch]
            if len(b.bearings) < 2:
                continue
            # 几何太差（近乎共线）时先补测，别盲目清除
            if b.direction_span_deg() < span_min:
                continue
            # 同一批示向度下最多尝试若干次清除，避免在低置信区域"网格式盲扫"
            tried = self.clear_attempts.get(ch, 0)
            mark = self.clear_tried_bearings.get(ch, -1)
            if len(b.bearings) == mark and tried >= max_attempts:
                continue
            pt, p_succ = self._best_clear_point(ch)
            if pt is None or p_succ < p_min:
                continue
            cost = self.cost.predict_clear(pt, ch, found=True)["total_s"]
            score = cost + w * (1.0 - p_succ)
            if best is None or score < best[0]:
                best = (score, ch, pt, p_succ, cost)
        if best is None:
            return None
        _, ch, pt, p_succ, cost = best
        self.clear_tried_bearings[ch] = len(self.world.beliefs[ch].bearings)
        return Action("clear", pt, ch, reason="localized", expected_cost_s=cost,
                      meta={"p_success": round(p_succ, 3),
                            "uncertainty_m": round(self.world.beliefs[ch].uncertainty_m(), 1)})

    def _best_clear_point(self, ch: int):
        """返回 (清除点, 该点一次成功的后验概率)。"""
        b = self.world.beliefs[ch]
        r = float(self.env["clear_radius_m"])
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
        cand = np.array([(x, y) for x in xs for y in ys], dtype=float)
        d = np.hypot(cand[:, None, 0] - pos[None, :, 0], cand[:, None, 1] - pos[None, :, 1])
        cnt = (d <= r).sum(axis=1)
        if cnt.size == 0:
            return None, 0.0
        i = int(np.argmax(cnt))
        return (float(cand[i, 0]), float(cand[i, 1])), float(cnt[i]) / float(pos.shape[0])

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
                else:
                    if self.logger:
                        self.logger.event("clear_miss", channel=action.channel,
                                          pos=action.position,
                                          attempts=self.clear_attempts[action.channel])

    def _probe_action(self) -> Optional[Action]:
        """信息型选点：在"探测哪个频道"和"去哪探测"上做统一的单位时间收益最大化。

        候选动作打分  score = value × gain / (cost + smooth)
            gain  = 拿到一条新示向度的概率
                    · 单示向度频道：问题二给出的第二点，gain = π(detect)
                    · 未见频道：gain = p_exist × P(该点能测到)
            value = 该探测对最终清除的贡献权重（第二点把"半成品"变成"可清除"，权重更高）
            cost  = 移动耗时 + 检测 5 s + 换频 1 s
        """
        w, c = self.world, self.cost
        cur = c.position
        smooth = float(self.pol.get("probe_cost_smooth_s", 10.0))
        v_discover = float(self.pol.get("value_discover", 0.6))
        v_second = float(self.pol.get("value_second_point", 1.0))
        speed = float(self.env["speed_mps"])
        dt_action = float(self.env["detect_s"])
        dt_switch = float(self.env["switch_s"])
        budget_s = float(self.cfg["budget"].get("real_time_safety_margin_s", 25.0)) * 0.0 + 1e9

        best = None  # (score, Action)

        def consider(score, action):
            nonlocal best
            if best is None or score > best[0]:
                best = (score, action)

        # 单频道测量次数上限：定向源无法被"证明不存在"，必须防止无限投入
        cap_all = int(self.pol.get("max_measures_per_channel", 10))
        budget_ok = {ch for ch in w.pending() if self.probe_count.get(ch, 0) < cap_all}
        local_allowed = self.measure_streak < self.measure_cap

        # ---- 候选 A：单示向度频道 → 问题二的第二检测点
        for ch in w.pending():
            if ch not in budget_ok:
                continue
            b = w.beliefs[ch]
            if len(b.bearings) != 1:
                continue
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
            P2 = plan["P2"]
            travel = math.hypot(P2[0] - cur[0], P2[1] - cur[1]) / speed
            cost = travel + dt_action + dt_switch
            gain = max(0.0, float(plan["pi"])) * float(b.p_exist)
            if gain <= 1e-6:
                continue
            consider(v_second * gain / (cost + smooth),
                     Action("measure", P2, ch, reason="q2_second_point",
                            expected_cost_s=cost,
                            meta={"E_D": round(plan["E_D"], 2), "pi": round(plan["pi"], 3),
                                  "d": round(plan["d"], 1), "beta": round(plan["beta"], 1)}))

        # ---- 候选 B：从未见过的频道 → 在候选探测点里选收益/代价最高的
        unseen = [ch for ch in w.pending() if len(w.beliefs[ch].bearings) == 0 and ch in budget_ok]
        dcap = int(self.pol.get("max_discover_probes_per_channel", 8))
        unseen = [ch for ch in unseen if self.probe_count.get(ch, 0) < dcap]
        if unseen:
            pts = list(self.probe_grid)
            if local_allowed:
                pts = [cur] + pts
            else:
                # 到达就地测量上限后必须离开当前位置（栅格里正好含当前点，要排掉）
                pts = [p for p in pts
                       if math.hypot(p[0] - cur[0], p[1] - cur[1]) > 30.0]
            if not pts:
                return best[1] if best else None
            pts_arr = np.asarray(pts, dtype=float)
            for ch in unseen:
                b = w.beliefs[ch]
                probs = b.detect_prob(pts_arr)
                pe = b.p_exist
                if pe <= 1e-4:
                    continue
                for i, P in enumerate(pts):
                    gain = pe * float(probs[i])
                    if gain <= 1e-4:
                        continue
                    travel = math.hypot(P[0] - cur[0], P[1] - cur[1]) / speed
                    cost = travel + dt_action + dt_switch
                    consider(v_discover * gain / (cost + smooth),
                             Action("measure", (float(P[0]), float(P[1])), ch,
                                    reason="discover", expected_cost_s=cost,
                                    meta={"p_exist": round(pe, 3),
                                          "p_detect": round(float(probs[i]), 3)}))

        # ---- 候选 C：已有 >=2 条示向度但还不能可靠清除 → 补一条示向度收紧定位
        v_refine = float(self.pol.get("value_refine", 0.8))
        r_min = float(self.env["recv_radius_min_m"])
        r_max = float(self.env["recv_radius_max_m"])
        refine_pts = ([cur] + list(self.probe_grid)) if local_allowed else list(self.probe_grid)
        for ch in w.pending():
            if ch not in budget_ok:
                continue
            b = w.beliefs[ch]
            if len(b.bearings) < 2:
                continue
            est = b.point_estimate()
            if est is None:
                continue
            dirs = [(est[0] - p[0], est[1] - p[1]) for p, _ in b.bearings]
            for P in refine_pts:
                d_est = math.hypot(P[0] - est[0], P[1] - est[1])
                if d_est < 250.0 or d_est > r_min:
                    continue                       # 太近没意义、太远可能测不到
                v = (est[0] - P[0], est[1] - P[1])
                nv = math.hypot(v[0], v[1]) or 1.0
                worst = 180.0
                for dv in dirs:
                    nd = math.hypot(dv[0], dv[1]) or 1.0
                    c = (v[0] * dv[0] + v[1] * dv[1]) / (nv * nd)
                    worst = min(worst, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
                if worst < float(self.pol.get("refine_min_new_angle_deg", 20.0)):
                    continue                       # 与已有示向度太接近，加了也白加
                p_det = min(1.0, max(0.0, (r_max - d_est) / (r_max - r_min)))
                travel = math.hypot(P[0] - cur[0], P[1] - cur[1]) / speed
                cost = travel + dt_action + dt_switch
                consider(v_refine * p_det / (cost + smooth),
                         Action("measure", (float(P[0]), float(P[1])), ch,
                                reason="refine", expected_cost_s=cost,
                                meta={"worst_angle": round(worst, 1),
                                      "d_est": round(d_est, 1)}))
        return best[1] if best else None

    def should_stop(self) -> Optional[str]:
        w = self.world
        if not w.pending():
            return "all_resolved"
        if len(self.world.channels) - len(w.cleared) <= len(w.absent(self.absent_thr)):
            return "all_resolved"
        return None


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
