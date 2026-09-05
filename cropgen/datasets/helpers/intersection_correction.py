from cropgen.datasets.helpers.polygon_separation import separate_polygons, Vector2D
from shapely.geometry import MultiPolygon
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from typing import Any, Literal, Sequence
import numpy as np
from shapely import Polygon, STRtree
from shapely.affinity import translate

from cropgen.ocr_units import OCRLine, OCRParagraph
from cropgen.transforms.transforms import (
    IntraparagraphTransform,
    line_group_equivalent_type,
)


def avoid_line_intersections(
    polygons: list[Polygon],
    *,
    delta: float = 0.5,
    binary_iterations: int = 32,
    contact_epsilon: float = 1e-8,
) -> list[Polygon]:
    print("Avoiding")

    n = len(polygons)

    LGinfo = LineGroupInfo.from_polygons(polygons)

    original_center = LGinfo.center

    cumulative_shifts: list[Vector2D] = [
        np.zeros(2, dtype=float) for _ in polygons
    ]  # ty: ignore[invalid-assignment]

    reading_dir = LGinfo.reading_direction

    if n <= 1:
        return polygons

    union_polygon = LineGroupInfo.polygon_union(polygons)

    union_centroid = np.array(
        [
            union_polygon.centroid.x,
            union_polygon.centroid.y,
        ],
        dtype=float,
    )

    order = list(range(len(polygons)))

    left_extent = np.empty(n, dtype=float)
    right_extent = np.empty(n, dtype=float)

    for i, polygon in enumerate(polygons):
        coordinates = np.asarray(
            polygon.exterior.coords,
            dtype=float,
        )[:, :2]

        projections = (coordinates - union_centroid) @ reading_dir

        left_extent[i] = projections.min()
        right_extent[i] = projections.max()

    pairwise_displacements = np.zeros(n, dtype=float)

    for i in range(1, n):
        true_gap = polygons[i - 1].distance(polygons[i])

        if true_gap < delta:
            overlap = right_extent[i - 1] - left_extent[i] + delta

            if overlap > 0:
                pairwise_displacements[i] = overlap

    raw_forward_push: np.ndarray = np.cumsum(pairwise_displacements)

    displacement_along_line: np.ndarray = raw_forward_push - np.mean(raw_forward_push)

    for k, i in enumerate(order):
        scalar_shift = displacement_along_line[k]

        if abs(scalar_shift) < 1e-12:
            continue

        movement: Vector2D = scalar_shift * reading_dir

        polygons[i] = translate(
            polygons[i],
            xoff=movement[0],
            yoff=movement[1],
        )

        cumulative_shifts[i] += movement

    for k in range(1, n):
        max_reversal: float = raw_forward_push[k]

        if max_reversal <= 1e-12:
            continue

        i = order[k]
        polygon = polygons[i]

        previous_polygons = [polygons[order[prev_k]] for prev_k in range(k)]
        tree = STRtree(previous_polygons)

        def intersects_previous(distance: float) -> bool:
            candidate = translate(
                polygon,
                xoff=-reading_dir[0] * distance,
                yoff=-reading_dir[1] * distance,
            )
            # Query index with GEOS intersection predicate in C
            return tree.query(candidate, predicate="intersects").size > 0

        if not intersects_previous(max_reversal):
            displacement = max_reversal
        else:
            lo, hi = 0, max_reversal

            for _ in range(binary_iterations):
                mid = 0.5 * (lo + hi)

                if intersects_previous(mid):
                    hi = mid
                else:
                    lo = mid

            displacement = max(0.0, lo - contact_epsilon)

        if displacement <= 0:
            continue

        movement = reading_dir * (-displacement)

        polygons[i] = translate(
            polygon,
            xoff=movement[0],
            yoff=movement[1],
        )

        cumulative_shifts[i] += movement

    new_center = LineGroupInfo.from_polygons(polygons).center

    for i in range(len(polygons)):
        polygons[i] = translate(
            polygons[i],
            xoff=original_center[0] - new_center[0],
            yoff=original_center[1] - new_center[1],
        )

    return polygons


def avoid_paragraph_intersections(
    polygon_groups: list[list[Polygon]],
    delta: float = 5,
    max_iterations: int = 100,
    damping: float = 0.5,
) -> list[list[Polygon]]:

    # we use convex hulls: we dont want intersections to be so fine-grained that a far line
    # from a paragraph could be interleaved in the space between the lines of another.
    union_hulls = [
        LineGroupInfo.polygon_union(group).convex_hull for group in polygon_groups
    ]

    _, shifts = separate_polygons(
        union_hulls,
        delta=delta,
        max_iterations=max_iterations,
        damping=damping,
    )

    for polygon_group, shift in zip(polygon_groups, shifts):
        for i in range(len(polygon_group)):
            polygon_group[i] = translate(polygon_group[i], shift[0], shift[1])

    return polygon_groups
