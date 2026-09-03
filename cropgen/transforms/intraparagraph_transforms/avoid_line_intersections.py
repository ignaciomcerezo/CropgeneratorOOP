from shapely.geometry import MultiPolygon
from cropgen.transforms.helpers.line_group_info import LineGroupInfo, Vector2D
from typing import Any, Literal, Sequence, cast
import numpy as np
from numpy.random import permutation
from PIL import Image
from shapely import Polygon, STRtree
from shapely.affinity import translate
import numpy.typing as npt

from cropgen.processing.line import Line
from cropgen.processing.paragraph import Paragraph
from cropgen.transforms.transforms import IntraparagraphTransform


class AvoidLineIntersections(IntraparagraphTransform):

    def __init__(
        self,
        delta: float = 0.5,
        max_iterations: int = 1000,
    ):
        self.delta = delta
        self.max_iterations = max_iterations

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph | Sequence[Line] | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        images, raw_polygons = self._extract_polygons_and_images(line_equivalent_group)
        polygon_list = list(raw_polygons)

        polygons, _ = self.call_polygons(
            polygon_list,
        )
        return images.copy(), polygons

    def call_polygons(
        self,
        polygons: list[Polygon],
        *,
        binary_iterations: int = 32,
        contact_epsilon: float = 1e-8,
    ) -> tuple[list[Polygon], list[Vector2D]]:

        n = len(polygons)

        LGinfo = LineGroupInfo.from_polygons(polygons)

        original_center = LGinfo.center

        cumulative_shifts: list[Vector2D] = [
            np.zeros(2, dtype=float) for _ in polygons
        ]  # ty: ignore[invalid-assignment]

        reading_dir = LGinfo.reading_direction

        if n <= 1:
            return polygons, cumulative_shifts

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

            if true_gap < self.delta:
                overlap = right_extent[i - 1] - left_extent[i] + self.delta

                if overlap > 0:
                    pairwise_displacements[i] = overlap

        raw_forward_push: np.ndarray = np.cumsum(pairwise_displacements)

        displacement_along_line: np.ndarray = raw_forward_push - np.mean(
            raw_forward_push
        )

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

        return polygons, cumulative_shifts
