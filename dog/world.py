"""世界模型：每频道一个粒子信念 + 一个解析的"无源检验"。

证据似然在本题是**指示函数**（误差有界 ±1°、接收半径有界 1000~1500 m、
定向覆盖是半平面），落在可行域内概率为 1、域外为 0，于是"贝叶斯更新"退化为
"按约束筛选粒子"，可统一处理：
    direction  —— 可探测 且 与示向度偏差 <= δ
    near       —— 可探测 且 距离 <= 5 m
    no_signal  —— 不可探测（无源 / 超半径 / 定向盲区 三义性自动体现）
    clear 未命中 —— 距离 > 20 m
    clear 成功  —— 距离 <= 20 m（随后该频道标记为已清除）

"该频道到底有没有源"用解析法而不是蒙特卡洛：一个 ±1° 的方位约束会把球面均匀
撒点几乎全部筛掉，蒙卡分辨不了这种量级的似然；而无源检验只需要距离信息，
可以网格化后精确求和（见 ExistenceTest）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import geom

Point = Tuple[float, float]


class ExistenceTest(object):
    """无源检验 + "某点还能不能测到"的概率，网格化后解析计算。

    对每个格点记录 d_min = 该点到所有探测点的最小距离。于是：
      似然          P(全部无信号 | 源在该格点) = P(R < d_min)
      还能测到      P(在 P 处测到 | 源在该格点, 已有观测)
                    = P(R ∈ [d(P,cell), d_min))  —— 只有在比 d_min 更近的点才有机会
    第二式是"不要把同一个点反复探测"的关键：在某点已测过之后，该点的 p_detect 恰为 0。

    定向源（Q4）下"无信号"可能由盲区解释，证据更弱：似然取 1/2 次方近似折扣
    （每个探测点最多有一半的角度能解释"测不到"）。
    """

    def __init__(self, cfg: Dict[str, Any], mode: str, n_r: int = 40, n_th: int = 48):
        env = cfg["env"]
        self.mode = mode
        self.arena_r = float(env["arena_radius_m"])
        self.r_min = float(env["recv_radius_min_m"])
        self.r_max = float(env["recv_radius_max_m"])
        rs = (np.arange(n_r) + 0.5) * self.arena_r / n_r
        ths = (np.arange(n_th) + 0.5) * 2.0 * math.pi / n_th
        rr, tt = np.meshgrid(rs, ths, indexing="ij")
        self.cells = np.stack([rr.ravel() * np.cos(tt.ravel()),
                               rr.ravel() * np.sin(tt.ravel())], axis=1)
        self.w = rr.ravel()                      # 面积元 ∝ r
        self.w = self.w / self.w.sum()
        self.d_min = np.full(self.cells.shape[0], float("inf"))
        self.R_bins = np.linspace(self.r_min, self.r_max, 11)
        self.cnt = np.zeros((self.cells.shape[0], self.R_bins.size), dtype=np.int16)
        self.seen_signal = False
        self._dirty = True
        self._lk = None
        self._post = None
        self._det_cache: Dict[Any, np.ndarray] = {}

    # ------------------------------------------------------------------ 更新
    def observe(self, kind: str, P: Point) -> None:
        if kind in ("direction", "near"):
            self.seen_signal = True
            return
        if kind == "no_signal":
            d = np.hypot(self.cells[:, 0] - P[0], self.cells[:, 1] - P[1])
            self.d_min = np.minimum(self.d_min, d)
            self.cnt += (d[:, None] <= self.R_bins[None, :]).astype(np.int16)
            self._dirty = True
            self._det_cache.clear()

    # ------------------------------------------------------------------ 似然
    def cell_likelihood(self) -> np.ndarray:
        if self._lk is None or self._dirty:
            if self.mode == "q3":
                # 全向：R >= d_min 必然被测到，故 R 只能在 [r_min, d_min) 内取值
                self._lk = np.clip((self.d_min - self.r_min) / (self.r_max - self.r_min), 0.0, 1.0)
            else:
                # 定向：某点无信号可能只是落在盲区（概率 1/2），故似然为
                # (1/B) Σ_b (1/2)^m_b，m_b = 该 R 档下覆盖了本格点的探测点数
                self._lk = np.power(0.5, self.cnt.astype(float)).mean(axis=1)
            self._dirty = False
        return self._lk

    def loglike(self) -> float:
        return float((self.w * self.cell_likelihood()).sum())

    def cell_posterior(self) -> np.ndarray:
        if self._post is None or self._dirty:
            num = self.w * self.cell_likelihood()
            s = num.sum()
            self._post = self.w if s <= 0 else num / s
        return self._post

    def detect_prob(self, pts: np.ndarray) -> np.ndarray:
        """P(在某点能测到该频道信号 | 目前全部观测)。pts:(M,2) -> (M,)"""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        key = (pts.shape, round(float(pts.sum()), 3), round(float((pts ** 2).sum()), 3))
        hit = self._det_cache.get(key)
        if hit is not None:
            return hit
        post = self.cell_posterior()
        d = np.hypot(pts[:, None, 0] - self.cells[None, :, 0],
                     pts[:, None, 1] - self.cells[None, :, 1])
        hi = np.minimum(self.d_min, self.r_max)             # R 的可行上界
        denom = hi - self.r_min                             # >0 才有剩余可能
        num = hi[None, :] - d
        p = np.where(denom[None, :] > 1e-9,
                     np.clip(num / np.maximum(denom, 1e-9)[None, :], 0.0, 1.0), 0.0)
        if self.mode == "q4":
            p = 0.5 * p                                     # 覆盖半平面只有 180°
        out = (p * post[None, :]).sum(axis=1)
        self._det_cache[key] = out
        return out


class ChannelBelief(object):
    """单频道信念。"""

    def __init__(self, channel: int, cfg: Dict[str, Any], mode: str, rng: np.random.Generator):
        env = cfg["env"]
        pol = cfg["policy"]
        self.channel = int(channel)
        self.cfg = cfg
        self.env = env
        self.mode = mode                      # 'q3'（全向）或 'q4'（混合）
        self.rng = rng
        self.arena_r = float(env["arena_radius_m"])
        self.r_min = float(env["recv_radius_min_m"])
        self.r_max = float(env["recv_radius_max_m"])
        self.delta = float(env["bearing_error_deg"])
        self.p_exist_prior = float(pol.get("prior_exist_prob", 0.65))
        self.p_omni_prior = 1.0 if mode == "q3" else 0.5

        self.cleared = False
        self.bearings: List[Tuple[Point, float]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.clear_misses: List[Point] = []   # 已尝试但未命中的清除点（避免原地重复尝试）
        self.n_probes = 0                     # 本频道被测量的次数（用于投入上限）
        self.exist = ExistenceTest(cfg, mode)
        self.n_src = 600
        self.src: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------ 粒子初始化
    def _init_src_cloud(self, P: Point) -> None:
        """首次测到示向度后，按"已测到"条件生成源位置粒子。"""
        n = self.n_src
        R = self.rng.uniform(self.r_min, self.r_max, n)
        theta_true = self.bearings[0][1] + self.rng.uniform(-self.delta, self.delta, n)
        # r 的先验 ∝ r（面积元），且 r <= min(R, 该方向弦长)
        u = self.rng.random(n)
        rmax = np.minimum(R, np.array([geom.chord_length(P, t, (0.0, 0.0), self.arena_r)
                                       for t in theta_true]))
        rmax = np.maximum(rmax, 1.0)
        r = rmax * np.sqrt(u)
        rad = np.radians(theta_true)
        pos = np.stack([P[0] + r * np.cos(rad), P[1] + r * np.sin(rad)], axis=1)
        omni = (np.ones(n, dtype=bool) if self.mode == "q3"
                else (self.rng.random(n) < self.p_omni_prior))
        alpha = self.rng.uniform(0.0, 2.0 * math.pi, n)
        keep = self._likelihood_direction(P, self.bearings[0][1], pos, R, omni, alpha)
        if int(keep.sum()) < 20:               # 极端情形下放宽
            keep = np.ones(n, dtype=bool)
        self.src = {"pos": pos[keep], "R": R[keep],
                    "omni": omni[keep], "alpha": alpha[keep]}

    # -------------------------------------------------------------- 似然筛选
    def _detectable(self, P: Point, pos: np.ndarray, R: np.ndarray,
                    omni: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        rel = pos - np.array(P)                       # 源相对检测点
        d = np.hypot(rel[:, 0], rel[:, 1])
        within = d <= R
        if omni.all():
            return within
        # 覆盖条件：检测点位于源辐射方向 α 的半平面内 ⇒ dot(P-S, u(α)) >= 0
        to_p = -rel
        proj = to_p[:, 0] * np.cos(alpha) + to_p[:, 1] * np.sin(alpha)
        return within & (omni | (proj >= 0.0))

    def _likelihood_direction(self, P, theta_hat, pos, R, omni, alpha) -> np.ndarray:
        det = self._detectable(P, pos, R, omni, alpha)
        rel = pos - np.array(P)
        d = np.hypot(rel[:, 0], rel[:, 1])
        far = d > float(self.env["near_radius_m"])
        brg = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
        diff = np.abs(((brg - theta_hat) + 180.0) % 360.0 - 180.0)
        return det & far & (diff <= self.delta + 1e-9)

    def _likelihood_no_signal(self, P, pos, R, omni, alpha) -> np.ndarray:
        return ~self._detectable(P, pos, R, omni, alpha)

    def _likelihood_near(self, P, pos, R, omni, alpha) -> np.ndarray:
        det = self._detectable(P, pos, R, omni, alpha)
        rel = pos - np.array(P)
        d = np.hypot(rel[:, 0], rel[:, 1])
        return det & (d <= float(self.env["near_radius_m"]) + 1e-9)

    def _likelihood_far(self, P, pos, radius: float) -> np.ndarray:
        rel = pos - np.array(P)
        return np.hypot(rel[:, 0], rel[:, 1]) > radius

    # ---------------------------------------------------------------- 更新
    def observe(self, kind: str, P: Point, svd_deg: Optional[float] = None,
                clear_ok: Optional[bool] = None) -> None:
        self.evidence.append({"kind": kind, "pos": P, "svd": svd_deg})
        if kind in ("direction", "near", "no_signal"):
            self.n_probes += 1
        self.exist.observe(kind, P)
        if kind == "direction":
            self.bearings.append((P, float(svd_deg)))
            if self.src is None:
                self._init_src_cloud(P)
            else:
                self._filter_src(self._likelihood_direction(
                    P, float(svd_deg), self.src["pos"], self.src["R"],
                    self.src["omni"], self.src["alpha"]))
        elif kind == "near":
            if self.src is not None:
                self._filter_src(self._likelihood_near(
                    P, self.src["pos"], self.src["R"], self.src["omni"], self.src["alpha"]))
        elif kind == "no_signal":
            if self.src is not None:
                self._filter_src(self._likelihood_no_signal(
                    P, self.src["pos"], self.src["R"], self.src["omni"], self.src["alpha"]))
        elif kind == "clear":
            if clear_ok:
                self.cleared = True
            else:
                self.clear_misses.append((float(P[0]), float(P[1])))
                if self.src is not None:
                    self._filter_src(self._likelihood_far(
                        P, self.src["pos"], float(self.env["clear_radius_m"])))

    def _filter_src(self, keep: np.ndarray) -> None:
        cur = int(self.src["pos"].shape[0])
        if cur == 0:
            return
        n_keep = int(keep.sum())
        if n_keep < 30:
            # 退化：在存活粒子周围抖动重采样（保持共 N 个粒子），再重新施加约束
            if n_keep == 0:
                idx = self.rng.integers(0, cur, size=min(30, cur))
                base = self.src["pos"][idx]
                scale = 60.0
            else:
                idx = np.where(keep)[0]
                base = self.src["pos"][idx]
                scale = max(10.0, float(np.std(base, axis=0).max()) * 0.5 + 5.0)
            n_new = self.n_src
            pick = self.rng.integers(0, base.shape[0], size=n_new)
            jitter = self.rng.normal(0.0, scale, size=(n_new, 2))
            self.src["pos"] = base[pick] + jitter
            self.src["R"] = self.src["R"][idx][pick]
            self.src["omni"] = self.src["omni"][idx][pick]
            self.src["alpha"] = self.src["alpha"][idx][pick]
            self._reapply_all_direction_constraints()
        else:
            for key in ("pos", "R", "omni", "alpha"):
                self.src[key] = self.src[key][keep]

    def _reapply_all_direction_constraints(self) -> None:
        for ev in self.evidence:
            if ev["kind"] == "direction":
                keep = self._likelihood_direction(ev["pos"], ev["svd"], self.src["pos"],
                                                  self.src["R"], self.src["omni"], self.src["alpha"])
                if keep.any():
                    for key in ("pos", "R", "omni", "alpha"):
                        self.src[key] = self.src[key][keep]

    # ---------------------------------------------------------------- 输出
    @property
    def p_exist(self) -> float:
        if self.cleared:
            return 0.0
        # 测到过信号（direction/near）⇒ 该频道必然有源
        if self.exist.seen_signal:
            return 1.0
        p0 = self.p_exist_prior
        l = max(1e-12, self.exist.loglike())
        num = p0 * l
        den = num + (1.0 - p0)
        return num / den if den > 0 else 1.0

    def detect_prob(self, pts: np.ndarray) -> np.ndarray:
        """某点能测到本频道信号的概率（用于信息型选点）。"""
        return self.exist.detect_prob(pts)

    def coverage_prob(self, P: Point) -> float:
        """检测点 P 落在该源"有效覆盖半平面"内的概率（按 α 后验）。

        这是档 1.0 的关键：定向源的"能不能测到" = 距离条件 × 覆盖条件，
        而粒子云里的 omni/α 就是覆盖条件的后验。全向模式恒为 1。
        """
        if self.mode == "q3" or self.src is None or self.src["pos"].shape[0] == 0:
            return 1.0
        pos = self.src["pos"]
        omni = self.src["omni"]
        al = self.src["alpha"]
        rel = np.array([P[0], P[1]], dtype=float)[None, :] - pos
        proj = rel[:, 0] * np.cos(al) + rel[:, 1] * np.sin(al)
        return float((omni | (proj >= 0.0)).mean())

    def region(self) -> Optional[geom.Region]:
        if len(self.bearings) < 2:
            return None
        return geom.region_from_bearings(self.bearings, self.delta)

    def point_estimate(self) -> Optional[Point]:
        reg = self.region()
        if reg is not None and reg.valid:
            return reg.centroid()
        if self.src is not None and self.src["pos"].shape[0] > 0:
            med = np.median(self.src["pos"], axis=0)
            return (float(med[0]), float(med[1]))
        return None

    def uncertainty_m(self) -> float:
        """不确定度（米）：有精确区域时用区域直径，否则用粒子云尺度。"""
        reg = self.region()
        if reg is not None:
            return reg.diameter()
        if self.src is None or self.src["pos"].shape[0] < 20:
            return float("inf")
        pos = self.src["pos"]
        med = np.median(pos, axis=0)
        d = np.hypot(pos[:, 0] - med[0], pos[:, 1] - med[1])
        return 2.0 * float(np.percentile(d, 90))

    def direction_span_deg(self) -> float:
        """已有示向度之间的最大交会夹角（判断几何是否可用）。"""
        if len(self.bearings) < 2:
            return 0.0
        angles = []
        for i in range(len(self.bearings)):
            for j in range(i + 1, len(self.bearings)):
                p1, t1 = self.bearings[i]
                p2, _ = self.bearings[j]
                b1 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                angles.append(geom.angle_diff_deg(t1, b1))
        return max(angles) if angles else 0.0

    def alpha_stats(self):
        """定向方向的后验（Q4）：存活粒子中定向源的角度均值与集中度。"""
        if self.src is None or self.mode == "q3":
            return None
        omni = self.src["omni"]
        if omni.size == 0 or omni.all():
            return None
        a = self.src["alpha"][~omni]
        if a.size < 10:
            return None
        c = float(np.mean(np.cos(a)))
        s = float(np.mean(np.sin(a)))
        return {"alpha_deg": math.degrees(math.atan2(s, c)) % 360.0,
                "concentration": math.hypot(c, s), "n": int(a.size)}


class World(object):
    def __init__(self, cfg: Dict[str, Any], mode: str = "q3", seed: Optional[int] = None):
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        lo = int(cfg["env"]["channel_min"])
        hi = int(cfg["env"]["channel_max"])
        self.channels = list(range(lo, hi + 1))
        self.pending_thr = float(cfg["policy"].get("absent_mass_threshold", 0.02))
        # 准度优先：一个频道在被"判无源"之前，必须已经在一个覆盖级站点集合上被测过
        # （否则远端的源会因为几处无信号就被误判掉）
        self.min_probes_before_absent = int(
            cfg["policy"].get("min_probes_before_absent", 8))
        self.beliefs: Dict[int, ChannelBelief] = {
            ch: ChannelBelief(ch, cfg, mode, self.rng) for ch in self.channels}
        self.cleared: List[int] = []

    # ------------------------------------------------------------------ 观测
    def observe(self, kind: str, position: Point, channel: int,
                svd_deg: Optional[float] = None, clear_ok: Optional[bool] = None) -> None:
        b = self.beliefs[int(channel)]
        b.observe(kind, position, svd_deg=svd_deg, clear_ok=clear_ok)
        if clear_ok and int(channel) not in self.cleared:
            self.cleared.append(int(channel))

    # ------------------------------------------------------------------ 查询
    def pending(self) -> List[int]:
        out = []
        for ch in self.channels:
            if ch in self.cleared:
                continue
            b = self.beliefs[ch]
            if b.p_exist > self.pending_thr or b.n_probes < self.min_probes_before_absent:
                out.append(ch)
        return out

    def absent(self, threshold: float) -> List[int]:
        return [ch for ch in self.channels
                if ch not in self.cleared and self.beliefs[ch].p_exist <= threshold]

    def ready_to_clear(self, max_diameter_m: float) -> List[int]:
        out = []
        for ch in self.pending():
            b = self.beliefs[ch]
            if len(b.bearings) >= 2 and b.uncertainty_m() <= max_diameter_m:
                out.append(ch)
        return out

    def near_hits(self) -> List[int]:
        return [ch for ch in self.channels
                if ch not in self.cleared
                and any(e["kind"] == "near" for e in self.beliefs[ch].evidence)]

    def stats(self) -> Dict[str, Any]:
        return {
            "cleared": len(self.cleared),
            "pending": len(self.pending()),
            "absent": len([ch for ch in self.channels
                           if self.beliefs[ch].p_exist <= self.pending_thr]),
        }
