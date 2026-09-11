"""寻路与测点序列优化（纯几何 + 组合优化，无网络、无信念逻辑）。

思想来源：Bayram, Vander Hook & Isler, "Gathering Bearing Data for Target
Localization", UMN CS TR 15-014, 2015。该文研究同一个双准则问题：
    min  cost(S) = Σ 旅行距离 + n·单次测量耗时
    s.t. ∀ w ∈ T, ∃ s_i,s_j ∈ S :  U(s_i,s_j,w) ≤ U*
不确定度取 GDOP 型
    U(s_1,s_2,w) = d(s_1,w)·d(s_2,w) / |sin ∠s_1 w s_2|
解的结构是：先按几何把测点摆好（论文 Algorithm 1 用"半径 R′ 圆上彼此 120° 的三点"），
再对测点做 TSP 巡游（GatherData）。

本题对应：机器人=机器狗，方位=/measure 的示向度，测量耗时=5 s(+1 s 换频)，
旅行=直线距离/5 m/s，U* 由"能否在 20 m 内清除"反推。
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


# --------------------------------------------------------------------- GDOP
def gdop(s1: Point, s2: Point, w: Point) -> float:
    """U(s1,s2,w) = d1·d2/|sin∠s1 w s2|（米²）；近共线时发散。"""
    v1 = (s1[0] - w[0], s1[1] - w[1])
    v2 = (s2[0] - w[0], s2[1] - w[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return float("inf")
    sinphi = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
    if sinphi < 1e-6:
        return float("inf")
    return n1 * n2 / sinphi


def best_pair_gdop(observations: Sequence[Point], w: Point,
                   exclude: Optional[int] = None) -> float:
    """已有观测点中最好一对的 GDOP；不足两点返回 inf。"""
    best = float("inf")
    n = len(observations)
    for i in range(n):
        if i == exclude:
            continue
        for j in range(i + 1, n):
            if j == exclude:
                continue
            u = gdop(observations[i], observations[j], w)
            if u < best:
                best = u
    return best


def crossing_angle_deg(a: Point, b: Point, w: Point) -> float:
    """观测点 a、b 在目标 w 处张开的交会角（度）。"""
    v1 = (a[0] - w[0], a[1] - w[1])
    v2 = (b[0] - w[0], b[1] - w[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def gdop_radius(u_star: float, sigma_rad: float) -> float:
    """论文 Algorithm 1 第 1 行的 R：测点若全落在 D(w,R) 之外则必然无法满足 U*。"""
    if sigma_rad <= 0:
        return float("inf")
    return math.sqrt(max(u_star, 0.0)) / (2.0 * sigma_rad)


# ------------------------------------------------- 下一个补测点（GDOP + 120° 原则）
def choose_refine_point(target: Point,
                        observations: Sequence[Point],
                        u_star: float,
                        r_guaranteed: float,
                        ring_fracs: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
                        angle_step_deg: float = 10.0,
                        n_best: int = 3) -> Optional[Point]:
    """在"以 target 为心、半径 r_guaranteed 以内"的环上选补测点。

    判据：加入该点后最好一对的 GDOP 最小；再在最优的若干名里挑"与已有观测
    方向差异最大"的（论文的三点 120° 原则）。
    """
    if not observations:
        return None
    cands: List[Tuple[float, Point]] = []
    steps = max(4, int(round(360.0 / angle_step_deg)))
    for frac in ring_fracs:
        rho = float(r_guaranteed) * float(frac)
        if rho < 50.0:
            continue
        for k in range(steps):
            th = math.radians(k * angle_step_deg)
            p = (target[0] + rho * math.cos(th), target[1] + rho * math.sin(th))
            u = min(gdop(o, p, target) for o in observations)
            cands.append((u, p))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    top = cands[:max(1, int(n_best))]
    best = None
    for u, p in top:
        worst = min(crossing_angle_deg(o, p, target) for o in observations)
        score = (min(worst, 180.0 - worst), -u)
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else cands[0][1]


# ------------------------------------------------------------------ 巡游排序
def _tour_cost(order: Sequence[int], pts: Sequence[Point], start: Point) -> float:
    if not order:
        return 0.0
    total = math.hypot(pts[order[0]][0] - start[0], pts[order[0]][1] - start[1])
    for a, b in zip(order, order[1:]):
        total += math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
    return total


def nearest_neighbor_order(pts: Sequence[Point], start: Point) -> List[int]:
    left = set(range(len(pts)))
    order: List[int] = []
    cur = start
    while left:
        nxt = min(left, key=lambda i: math.hypot(pts[i][0] - cur[0], pts[i][1] - cur[1]))
        order.append(nxt)
        left.discard(nxt)
        cur = pts[nxt]
    return order


def two_opt(order: Sequence[int], pts: Sequence[Point], start: Point,
            max_pass: int = 3) -> List[int]:
    o = list(order)
    best = _tour_cost(o, pts, start)
    for _ in range(max_pass):
        improved = False
        n = len(o)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = o[:i] + list(reversed(o[i:j + 1])) + o[j + 1:]
                c = _tour_cost(cand, pts, start)
                if c + 1e-9 < best:
                    o, best, improved = cand, c, True
        if not improved:
            break
    return o


def tour_order(pts: Sequence[Point], start: Point) -> Tuple[List[int], float]:
    """返回 (访问顺序, 总里程)。n≤7 暴力最优；否则最近邻 + 2-opt。

    （实测：再加 Or-opt 只省 0.5% 里程却掉 0.8pp 清除比例，故不采用；
      持久巡游/最廉插入与"每步重规划"实测无差别，同样不采用。）
    """
    n = len(pts)
    if n == 0:
        return [], 0.0
    if n <= 7:
        best, best_cost = None, float("inf")
        for perm in itertools.permutations(range(n)):
            c = _tour_cost(perm, pts, start)
            if c < best_cost:
                best, best_cost = list(perm), c
        return (best or []), best_cost
    o = two_opt(nearest_neighbor_order(pts, start), pts, start)
    return o, _tour_cost(o, pts, start)


# ------------------------------------------------- U* 与"定位区域直径"的换算
def region_diameter_from_gdop(u: float, delta_rad: float) -> float:
    """正交几何下 D = 2√2·δ·r 与 U = r² 的换算（r 为源距），用于设定 U*。"""
    if u <= 0:
        return 0.0
    return 2.0 * math.sqrt(2.0) * delta_rad * math.sqrt(u)


def gdop_from_region_diameter(d: float, delta_rad: float) -> float:
    """上式的反函数：由"允许的区域直径"反推 U*。"""
    if d <= 0 or delta_rad <= 0:
        return 0.0
    r = d / (2.0 * math.sqrt(2.0) * delta_rad)
    return r * r


# ------------------------------------------------------- 覆盖站点（PlaceSensors）
def hex_grid(arena_r: float, spacing: float, offset: Tuple[float, float] = (0.0, 0.0),
             center: Tuple[float, float] = (0.0, 0.0)) -> List[Point]:
    """圆盘内的六边形栅格。"""
    pts: List[Point] = []
    dy = spacing * math.sqrt(3.0) / 2.0
    ny = int(arena_r / dy) + 2
    nx = int(arena_r / spacing) + 2
    for iy in range(-ny, ny + 1):
        y = iy * dy + offset[1]
        off = (spacing / 2.0) if (iy % 2) else 0.0
        for ix in range(-nx, nx + 1):
            x = ix * spacing + off + offset[0]
            if math.hypot(x - center[0], y - center[1]) <= arena_r:
                pts.append((x, y))
    return pts


def greedy_cover(arena_r: float, cover_radius: float,
                n_r: int = 20, n_th: int = 24) -> List[Point]:
    """用贪心集合覆盖求"覆盖整个圆盘所需的最少站点"（任意覆盖半径都适用）。

    栅格化的六边形布点在覆盖半径较大时会退化（间距超过圆盘直径后只剩圆心一个点），
    贪心覆盖则始终给出接近最优的站数——这决定了发现阶段的固定里程。
    """
    cells: List[Point] = []
    for i in range(n_r):
        r = (i + 0.5) * arena_r / n_r
        for j in range(n_th):
            th = 2.0 * math.pi * (j + 0.5) / n_th
            cells.append((r * math.cos(th), r * math.sin(th)))
    n = len(cells)
    r2 = cover_radius * cover_radius
    covered = [False] * n
    centers: List[Point] = []
    while not all(covered):
        best_i, best_c = -1, -1
        for i in range(n):
            if covered[i]:
                continue
            cx, cy = cells[i]
            c = 0
            for j in range(n):
                if not covered[j]:
                    continue
                dx = cx - cells[j][0]; dy = cy - cells[j][1]
                if dx * dx + dy * dy <= r2:
                    c += 1
            if c > best_c:
                best_c, best_i = c, i
        if best_i < 0:
            break
        cx, cy = cells[best_i]
        centers.append((cx, cy))
        for j in range(n):
            if covered[j]:
                continue
            dx = cx - cells[j][0]; dy = cy - cells[j][1]
            if dx * dx + dy * dy <= r2:
                covered[j] = True
    return centers


def covering_stations(arena_r: float, cover_radius: float,
                      stagger: bool = True) -> List[Point]:
    """覆盖站点：任意点都落在某个站点的 cover_radius 之内（六边形覆盖半径 = s/√3）。

    stagger=True 时再生成一组"错位半间距"的站点（落在第一组栅格的空洞处）。
    返回 (pass1, pass2)：pass1 保证"任意源必被某站测到"；pass2 只为让更多源能
    拿到第二条示向度，因此**可以按需展开**（见 TourPlanner.unvisited_stations）。
    """
    s = cover_radius * math.sqrt(3.0)
    pass1 = hex_grid(arena_r, s)
    pass2 = hex_grid(arena_r, s, offset=(s / 2.0, s * math.sqrt(3.0) / 6.0)) if stagger else []
    return pass1, pass2
