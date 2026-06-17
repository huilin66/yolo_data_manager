from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class XYXY:
    x1: float
    y1: float
    x2: float
    y2: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def xywhn_to_xyxy(box: Sequence[float], width: int, height: int) -> XYXY:
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return XYXY(x1, y1, x2, y2)


def xyxy_to_xywhn(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def polygon_to_box(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, x_max - x_min, y_max - y_min


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def normalized_points_to_pixels(
    points: Iterable[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    return [(x * width, y * height) for x, y in points]


def polygon_self_intersects(points: Sequence[tuple[float, float]]) -> bool:
    if len(points) < 4:
        return False
    edges = list(zip(points, points[1:] + points[:1]))
    for idx, edge_a in enumerate(edges):
        for jdx, edge_b in enumerate(edges):
            if abs(idx - jdx) <= 1 or {idx, jdx} == {0, len(edges) - 1}:
                continue
            if _segments_intersect(edge_a[0], edge_a[1], edge_b[0], edge_b[1]):
                return True
    return False


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
