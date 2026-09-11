"""问题一几何内核（运行时复用 + 论文出图共用一套实现）。

模型：
    检测点 P_i 的示向度 θ_i 与误差界 ±δ 确定一个扇形（= 两个半平面之交）；
    n 个检测点的定位区域 R = ∩ 扇形_i = { x : A x <= b }，是凸多边形；
    直径 D = 区域内任意两点最大距离 = 凸多边形顶点对最大距离；
    覆盖判定：以线段 ab 为直径的圆覆盖 R  <=>  所有顶点 v 满足 (v-a)·(v-b) <= 0。

退化情形：
    空集   —— 测量彼此不相容（正常建模下真源必在两扇形内，故两站时不会为空）；
    无界   —— 各示向度方向区间 [θ_i-δ, θ_i+δ] 有公共方向（示向线近平行/反向）。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
HalfPlane = Tuple[Point, Point]  # (q, n)：约束 n·(x - q) >= 0

DEFAULT_BOX = 1.0e6
EPS = 1e-9
UNBOUNDED_DIAMETER = float("inf")


# --------------------------------------------------------------------- 基础
def unit(deg: float) -> Point:
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def perp(a: Point) -> Point:
    return (-a[1], a[0])


def normalize_deg(deg: float) -> float:
    return deg % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """两方位角的最小夹角，结果在 [0,180]。"""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


# ------------------------------------------------------------------ 半平面
def wedge_halfplanes(p: Point, theta_deg: float, delta_deg: float) -> List[HalfPlane]:
    """示向度 theta ± delta 的扇形，用两个半平面表示（内部满足 n·(x-p) >= 0）。"""
    dm = unit(theta_deg - delta_deg)
    dp = unit(theta_deg + delta_deg)
    n1 = perp(dm)                       # cross(dm, x-p) >= 0
    pd = perp(dp)
    n2 = (-pd[0], -pd[1])               # cross(dp, x-p) <= 0
    return [(p, n1), (p, n2)]


def disc_halfplanes(center: Point, radius: float, segments: int = 64) -> List[HalfPlane]:
    """用 segments 边形近似圆盘（凸），其边半平面的交即为该近似域约束。"""
    hps = []
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        v0 = (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0))
        v1 = (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1))
        e = sub(v1, v0)
        n = perp(e)                     # 逆时针顶点 ⇒ 内法向为边的左手法向
        # 使约束在圆心处成立
        if dot(n, sub(center, v0)) < 0:
            n = (-n[0], -n[1])
        hps.append((v0, n))
    return hps


def clip(poly: List[Point], hp: HalfPlane, eps: float = EPS) -> List[Point]:
    """Sutherland–Hodgman：用半平面 n·(x-q) >= 0 裁剪凸多边形。"""
    if len(poly) < 3:
        return []
    q, n = hp
    out: List[Point] = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        fa = dot(n, sub(a, q))
        fb = dot(n, sub(b, q))
        if fa >= -eps:
            out.append(a)
        if (fa > eps and fb < -eps) or (fa < -eps and fb > eps):
            t = fa / (fa - fb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return _tidy(out, eps)


def _tidy(poly: List[Point], eps: float = EPS) -> List[Point]:
    res: List[Point] = []
    for p in poly:
        if not res or dist(p, res[-1]) > 1e-6:
            res.append(p)
    if len(res) > 2 and dist(res[0], res[-1]) <= 1e-6:
        res.pop()
    return res if len(res) >= 3 else []


class Region(object):
    """定位区域：凸多边形。empty 表示不相容；bounded=False 表示无界（直径发散）。"""

    __slots__ = ("vertices", "bounded", "empty")

    def __init__(self, vertices: List[Point], bounded: bool = True, empty: bool = False):
        self.vertices = vertices
        self.bounded = bounded
        self.empty = empty

    @property
    def valid(self) -> bool:
        return (not self.empty) and self.bounded and len(self.vertices) >= 3

    def diameter(self) -> float:
        if not self.valid:
            return UNBOUNDED_DIAMETER if not self.empty else 0.0
        return diameter(self.vertices)[0]

    def centroid(self) -> Optional[Point]:
        if len(self.vertices) < 3:
            return None
        sx = sum(p[0] for p in self.vertices) / len(self.vertices)
        sy = sum(p[1] for p in self.vertices) / len(self.vertices)
        return (sx, sy)

    def __repr__(self) -> str:
        return "Region(n=%d, bounded=%s, empty=%s)" % (len(self.vertices), self.bounded, self.empty)


def direction_intervals_intersect(thetas_deg: Sequence[float], delta_deg: float) -> bool:
    """各方向区间 [θ_i-δ, θ_i+δ] 是否有公共方向（等价于定位区域无界）。"""
    arcs = [((t - delta_deg) % 360.0, (t + delta_deg) % 360.0) for t in thetas_deg]
    if not arcs:
        return False
    for start, _ in arcs:
        if all(_in_arc(start, s, e) for (s, e) in arcs):
            return True
    return False


def _in_arc(x: float, start: float, end: float, eps: float = 1e-9) -> bool:
    """角度 x 是否落在从 start 逆时针到 end 的闭弧内（弧长 <= 360）。"""
    if start <= end:
        return (start - eps) <= x <= (end + eps)
    return x >= (start - eps) or x <= (end + eps)


def region_from_halfplanes(halfplanes: Sequence[HalfPlane],
                           box: float = DEFAULT_BOX) -> Region:
    """逐条裁剪求交。若中途为空 ⇒ 不相容；若顶点触盒 ⇒ 无界。"""
    poly: List[Point] = [(-box, -box), (box, -box), (box, box), (-box, box)]
    for hp in halfplanes:
        poly = clip(poly, hp)
        if not poly:
            return Region([], bounded=True, empty=True)
    bounded = not any(abs(p[0]) >= box - 1.0 or abs(p[1]) >= box - 1.0 for p in poly)
    return Region(poly, bounded=bounded, empty=False)


def region_from_bearings(stations: Sequence[Tuple[Point, float]],
                         delta_deg: float,
                         extra_halfplanes: Optional[Sequence[HalfPlane]] = None,
                         box: float = DEFAULT_BOX) -> Region:
    """stations = [(P_i, θ_i)]，返回定位区域。"""
    thetas = [t for _, t in stations]
    if len(stations) >= 2 and direction_intervals_intersect(thetas, delta_deg):
        return Region([], bounded=False, empty=False)
    hps: List[HalfPlane] = []
    for p, t in stations:
        hps.extend(wedge_halfplanes(p, t, delta_deg))
    if extra_halfplanes:
        hps.extend(extra_halfplanes)
    return region_from_halfplanes(hps, box=box)


def region_from_sector_intersections(stations: Sequence[Tuple[Point, float]],
                                     delta_deg: float) -> List[Point]:
    """两站的闭式解（正弦定理四角），用于与逐条裁剪互验。"""
    (p1, t1), (p2, t2) = stations[0], stations[1]
    pts: List[Point] = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            x = _line_intersection(p1, unit(t1 + s1 * delta_deg), p2, unit(t2 + s2 * delta_deg))
            if x is None:
                continue
            if _in_wedge(p1, t1, delta_deg, x) and _in_wedge(p2, t2, delta_deg, x):
                if all(dist(x, y) > 1e-9 for y in pts):
                    pts.append(x)
    return pts


def _line_intersection(p1: Point, d1: Point, p2: Point, d2: Point) -> Optional[Point]:
    den = cross(d1, d2)
    if abs(den) < 1e-15:
        return None
    w = sub(p2, p1)
    t = cross(w, d2) / den
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _in_wedge(p: Point, theta_deg: float, delta_deg: float, x: Point) -> bool:
    v = sub(x, p)
    return (cross(unit(theta_deg - delta_deg), v) >= -1e-9
            and cross(unit(theta_deg + delta_deg), v) <= 1e-9)


# -------------------------------------------------------------------- 指标
def diameter(poly: Sequence[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """凸多边形直径：顶点对最大距离（暴力 O(m^2)，m <= 2n 足够）。"""
    best = 0.0
    pair = None
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(poly[i], poly[j])
            if d > best:
                best = d
                pair = (poly[i], poly[j])
    return best, pair


def covered_by_diameter_circle(poly: Sequence[Point],
                               a: Point, b: Point,
                               tol: float = 1e-6) -> bool:
    """Thales 判据：(v-a)·(v-b) <= 0 <=> 顶点 v 落在以 ab 为直径的圆内。"""
    return all(dot(sub(v, a), sub(v, b)) <= tol for v in poly)


def min_enclosing_circle(poly: Sequence[Point]) -> Tuple[Point, float]:
    """最小包围圆（枚举 2 点/3 点定圆，O(m^3)，m 很小）。"""
    pts = list(poly)
    best = None  # (r, c)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            c = ((pts[i][0] + pts[j][0]) / 2.0, (pts[i][1] + pts[j][1]) / 2.0)
            r = dist(c, pts[i])
            if all(dist(c, p) <= r + 1e-6 for p in pts):
                if best is None or r < best[0] - 1e-9:
                    best = (r, c)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = _circumcenter(pts[i], pts[j], pts[k])
                if cc is None:
                    continue
                r = dist(cc, pts[i])
                if all(dist(cc, p) <= r + 1e-6 for p in pts):
                    if best is None or r < best[0] - 1e-9:
                        best = (r, cc)
    if best is None:
        return (pts[0] if pts else (0.0, 0.0)), 0.0
    return best[1], best[0]


def _circumcenter(a: Point, b: Point, c: Point) -> Optional[Point]:
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2 = a[0] * a[0] + a[1] * a[1]
    b2 = b[0] * b[0] + b[1] * b[1]
    c2 = c[0] * c[0] + c[1] * c[1]
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    return (ux, uy)


def chord_length(p: Point, theta_deg: float, center: Point = (0.0, 0.0),
                 radius: float = 1800.0) -> float:
    """从 p 沿 theta 方向的射线与圆盘的交点到 p 的距离。"""
    u = unit(theta_deg)
    w = sub(p, center)
    b = dot(w, u)
    c = dot(w, w) - radius * radius
    disc = b * b - c
    if disc <= 0:
        return 0.0
    return -b + math.sqrt(disc)
