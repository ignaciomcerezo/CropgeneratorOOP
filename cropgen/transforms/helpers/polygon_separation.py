from typing import Literal, Any
import numpy as np
from shapely import Polygon
from shapely.affinity import translate

Vector2D = np.ndarray[tuple[Literal[2], Any]]


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
