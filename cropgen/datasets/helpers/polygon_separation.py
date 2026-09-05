from cropgen.shared.parameters import Vector2D
from typing import Literal, Any
import numpy as np
from shapely import Polygon
from shapely.affinity import translate


def _polygon_axes(coords: np.ndarray) -> list[np.ndarray]:
    """Outward edge normals of a convex polygon: one candidate SAT axis per edge."""
    axes: list[np.ndarray] = []
    n = len(coords)
    for i in range(n):
        edge = coords[(i + 1) % n] - coords[i]
        normal = np.array([-edge[1], edge[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            axes.append(normal / norm)
    return axes


def _coords_of(polygon: Polygon) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def sat_minimum_translation_vector(
    poly_a: Polygon,
    poly_b: Polygon,
) -> Vector2D | None:
    """
    Minimal translation to apply to poly_b so it no longer overlaps
    poly_a.
    """
    coords_a = _coords_of(poly_a)
    coords_b = _coords_of(poly_b)

    axes = _polygon_axes(coords_a) + _polygon_axes(coords_b)
    if not axes:
        return None

    min_overlap = np.inf
    min_axis: Vector2D | None = None
    push_sign = 1.0

    for axis in axes:
        proj_a = coords_a @ axis
        proj_b = coords_b @ axis

        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        overlap = min(max_a, max_b) - max(min_a, min_b)

        if overlap <= 0:
            # Found a separating axis: the polygons don't actually overlap.
            return None

        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis
            center_a = 0.5 * (min_a + max_a)
            center_b = 0.5 * (min_b + max_b)
            push_sign = 1.0 if center_b >= center_a else -1.0

    assert min_axis is not None
    return min_axis * min_overlap * push_sign


def separate_polygons(
    polygons: list[Polygon],
    *,
    delta: float = 0.5,
    max_iterations: int = 1000,
    damping: float = 0.5,
) -> tuple[list[Polygon], list[Vector2D]]:
    n = len(polygons)
    shifts: list[Vector2D] = [
        np.zeros(2, dtype=float) for _ in range(n)
    ]  # ty: ignore[invalid-assignment]

    if n <= 1:
        return list(polygons), shifts

    current = list(polygons)

    for _ in range(max_iterations):
        moves = [np.zeros(2, dtype=float) for _ in range(n)]
        any_adjustment = False

        for i in range(n):
            for j in range(i + 1, n):
                mtv = sat_minimum_translation_vector(current[i], current[j])

                if mtv is None:
                    gap = current[i].distance(current[j])
                    if gap >= delta:
                        continue
                    centroid_i = np.array(
                        [current[i].centroid.x, current[i].centroid.y]
                    )
                    centroid_j = np.array(
                        [current[j].centroid.x, current[j].centroid.y]
                    )
                    direction = centroid_j - centroid_i
                    norm = np.linalg.norm(direction)
                    direction = (
                        direction / norm if norm > 1e-12 else np.array([1.0, 0.0])
                    )
                    push = (delta - gap) * direction
                else:
                    # Pad the raw SAT overlap by delta so shapes end up
                    # with real clearance, not just zero overlap.
                    unit = mtv / max(np.linalg.norm(mtv), 1e-12)
                    push = mtv + delta * unit

                half = 0.5 * damping * push
                moves[j] += half
                moves[i] -= half
                any_adjustment = True

        if not any_adjustment:
            break

        for i in range(n):
            if np.any(moves[i]):
                current[i] = translate(current[i], xoff=moves[i][0], yoff=moves[i][1])
                shifts[i] += moves[i]

    return current, shifts


import numpy as np
from shapely.affinity import translate


def separate_polygons(
    polygons: list[Polygon],
    *,
    delta: float = 5,
    max_iterations: int = 100,
    damping: float = 0.5,
    tol: float = 1e-3,
) -> tuple[list[Polygon], list[Vector2D]]:
    n = len(polygons)
    shifts: list[Vector2D] = [
        np.zeros(2, dtype=float) for _ in range(n)
    ]  # ty: ignore[invalid-assignment]

    if n <= 1:
        return list(polygons), shifts

    current = list(polygons)

    # Cache AABBs so we don't re-derive them from geometry every iteration;
    # we can update them exactly (translate is a pure shift) as we move things.
    bounds = np.array(
        [p.bounds for p in current], dtype=float
    )  # (n, 4): minx, miny, maxx, maxy

    for _ in range(max_iterations):
        moves = [np.zeros(2, dtype=float) for _ in range(n)]
        any_adjustment = False
        max_move_sq = 0.0

        # --- Broad phase: vectorized AABB check, padded by delta ---
        # Two polygons can only possibly need work (overlap OR gap < delta)
        # if their bounding boxes are within `delta` of each other. Since a
        # polygon's AABB always contains it, this can never falsely skip a
        # real candidate pair — it only skips pairs that are provably too
        # far apart, which is exactly what the old code's `continue` did,
        # just without paying for SAT/distance to find out.
        minx, miny, maxx, maxy = bounds.T
        overlap_x = (minx[:, None] - delta <= maxx[None, :]) & (
            maxx[:, None] + delta >= minx[None, :]
        )
        overlap_y = (miny[:, None] - delta <= maxy[None, :]) & (
            maxy[:, None] + delta >= miny[None, :]
        )
        candidates = np.triu(overlap_x & overlap_y, k=1)
        iu, ju = np.where(candidates)

        for i, j in zip(iu.tolist(), ju.tolist()):
            mtv = sat_minimum_translation_vector(current[i], current[j])

            if mtv is None:
                gap = current[i].distance(current[j])
                if gap >= delta:
                    continue
                centroid_i = np.array([current[i].centroid.x, current[i].centroid.y])
                centroid_j = np.array([current[j].centroid.x, current[j].centroid.y])
                direction = centroid_j - centroid_i
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 1e-12 else np.array([1.0, 0.0])
                push = (delta - gap) * direction
            else:
                unit = mtv / max(np.linalg.norm(mtv), 1e-12)
                push = mtv + delta * unit

            half = 0.5 * damping * push
            moves[j] += half
            moves[i] -= half
            any_adjustment = True

        if not any_adjustment:
            break

        for i in range(n):
            mv = moves[i]
            mv_sq = mv[0] * mv[0] + mv[1] * mv[1]
            if mv_sq > 1e-18:  # skip rebuilding geometry for truly negligible shifts
                current[i] = translate(current[i], xoff=mv[0], yoff=mv[1])
                shifts[i] += mv
                bounds[i, 0] += mv[0]
                bounds[i, 2] += mv[0]
                bounds[i, 1] += mv[1]
                bounds[i, 3] += mv[1]
                if mv_sq > max_move_sq:
                    max_move_sq = mv_sq

        # Stop once nobody is moving meaningfully instead of chasing exact
        # zero, which the damped update rarely reaches in floating point.
        if max_move_sq < tol * tol:
            break

    return current, shifts
