from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from typing import Any, Literal, Sequence, cast
import numpy as np
from numpy.random import permutation
from PIL import Image
from shapely import Polygon, STRtree
from shapely.affinity import translate

from cropgen.processing.line import Line
from cropgen.processing.paragraph import Paragraph
from cropgen.transforms.transforms import IntraparagraphTransform

Vector2D = np.ndarray[tuple[Literal[2], Any]]


class AvoidLineIntersections(IntraparagraphTransform):

    def __init__(
        self,
        delta: float = 0.5,
        max_iterations: int = 1000,
        only_reading_direction: bool = True,
    ):
        self.delta = delta
        self.max_iterations = max_iterations
        self.only_reading_direction = only_reading_direction

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph | Sequence[Line] | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        images, raw_polygons = self._extract_polygons_and_images(line_equivalent_group)
        polygon_list = list(raw_polygons)

        info = LineGroupInfo.from_polygons(polygon_list)

        polygons, _ = self.call_polygons(
            polygon_list,
            info.avg_rotation,
            self.only_reading_direction,
        )
        return images.copy(), polygons

    def call_polygons(
        self,
        polygons: list[Polygon],
        rot: float | Sequence[float] | None = None,
        project_to_reading_direction: bool = True,
        binary_iterations=32,
        contact_epsilon=1e-8,
    ) -> tuple[list[Polygon], list[Vector2D]]:

        n = len(polygons)

        original_center = LineGroupInfo.from_polygons(polygons).center

        cumulative_shifts: list[Vector2D] = [
            np.zeros(2, dtype=float) for _ in polygons
        ]  # ty: ignore[invalid-assignment]

        if n <= 1:
            return polygons, cumulative_shifts

        if not project_to_reading_direction or rot is None:
            return polygons, cumulative_shifts

        if isinstance(rot, (int, float)):
            angle = np.radians(rot)
        else:
            angles = np.radians(np.asarray(rot, dtype=float))
            angle = np.mean(angles)

        # Coordinates are y-down (image/document convention), so a reading
        # direction that is visually 90 degrees counterclockwise from the
        # horizontal rotation corresponds to angle - pi/2 in cos/sin terms,
        # not angle + pi/2.
        reading_dir = np.array(
            [np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)],
            dtype=float,
        )

        union_polygon = LineGroupInfo.polygon_union(polygons)

        union_centroid = np.array(
            [
                union_polygon.centroid.x,
                union_polygon.centroid.y,
            ],
            dtype=float,
        )

        centers = np.array(
            [[polygon.centroid.x, polygon.centroid.y] for polygon in polygons],
            dtype=float,
        )

        centroid_coordinates = (centers - union_centroid) @ reading_dir

        order = np.argsort(centroid_coordinates)

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

        for k in range(1, n):
            previous = order[k - 1]
            current = order[k]

            overlap = right_extent[previous] - left_extent[current] + self.delta

            if overlap > 0:
                pairwise_displacements[k] = overlap

        displacement_along_line = np.cumsum(pairwise_displacements)

        # we center the displacement field so the group does not drift in the reading direction
        displacement_along_line -= np.mean(displacement_along_line)

        for k, i in enumerate(order):
            scalar_shift = displacement_along_line[k]

            if abs(scalar_shift) < 1e-12:
                continue

            movement = scalar_shift * reading_dir

            polygons[i] = translate(
                polygons[i],
                xoff=movement[0],
                yoff=movement[1],
            )

            cumulative_shifts[i] += movement

        for k in range(1, n):
            i = order[k]

            polygon = polygons[i]

            def intersects_previous(distance: float) -> bool:
                candidate = translate(
                    polygon,
                    xoff=-reading_dir[0] * distance,
                    yoff=-reading_dir[1] * distance,
                )

                for previous_k in range(k):
                    j = order[previous_k]

                    if candidate.intersects(polygons[j]):
                        return True

                return False

            hi = max(self.delta, 1e-6)

            while not intersects_previous(hi):
                hi *= 2.0

                if hi > 1e6:
                    break

            if hi > 1e6:
                continue

            lo = 0.0

            for _ in range(binary_iterations):
                mid = 0.5 * (lo + hi)

                if intersects_previous(mid):
                    hi = mid
                else:
                    lo = mid

            displacement = max(0.0, lo - contact_epsilon)

            if displacement <= 0:
                continue

            movement = -displacement * reading_dir

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

    @staticmethod
    def _poly_centroid_asarray(polygon: Polygon) -> Vector2D:
        return np.array([polygon.centroid.x, polygon.centroid.y], dtype=float)
