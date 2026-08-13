from cropgen.processing.line import Line
from typing import Optional, Iterator

import numpy as np
from PIL import Image
from shapely import coverage_union_all
from shapely import Polygon
from shapely.affinity import affine_transform

from cropgen.processing.helpers.helper_to_classes import (
    is_path_graph,
)


class Paragraph:
    __slots__ = (
        "lines",
        "line_ids",
        "centroid",
        "total_words",
        "avg_rotation",
        "top",
        "left",
        "right",
        "bot",
        "task_id",
        "index",
        "subgraph",
    )

    def __init__(
        self,
        lines: list[Line] | None = None,
        task_id: int | None = None,
        index: int | None = None,
        subgraph: dict[str, set[str]] | None = None,
    ):

        if not lines:
            raise ValueError("Lines cannot be empty.")

        self.lines = lines
        self.task_id: int | None = task_id
        self.index: int | None = index
        self.subgraph: Optional[dict[str, set[str]]] = subgraph

        self._calculate_total_area_and_centroid()

        self._sort_lines_using_centroid_and_subgraph()

        self.line_ids = [line.id for line in self.lines]

    def __iter__(self) -> Iterator[Line]:
        for x in self.lines:
            yield x

    def __getitem__(self, index) -> Line:
        return self.lines[index]

    def __lt__(self, other: "Paragraph"):
        return (self.top, self.left) < (other.top, other.left)

    def __gt__(self, other: "Paragraph"):
        return (self.top, self.left) > (other.top, other.left)

    def transcription(self, separator: str = " "):
        return separator.join([fragment.text for fragment in self.lines])

    def __len__(self):
        return len(self.line_ids)

    def __repr__(self):
        return f"<{self.index}-th paragraph of order {len(self)} contained in AnnotatedPage of task ({self.task_id})>"

    def union_polygon(self) -> Polygon:
        return coverage_union_all([line.polygon for line in self.lines])

    def corrected_polygon(self, line: Line):
        t: float = np.radians(self.avg_rotation)
        a: float = np.cos(t)
        b: float = -np.sin(t)
        c: float = np.sin(t)
        d: float = np.cos(t)
        x_c: float = -float(self.centroid[0])
        y_c: float = -float(self.centroid[1])
        return affine_transform(line.polygon, [a, b, c, d, -x_c, -y_c])

    @staticmethod
    def _get_average_rotation(
        angles_in_degrees: list[float], areas: list[float]
    ) -> float:

        angles_in_radians = np.radians(angles_in_degrees)
        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))
        return -float(np.degrees(np.arctan2(sum_sin, sum_cos)))

    def generate_connected_subgraphs(
        self, order: int, max_subgraphs_to_generate: Optional[int] = None
    ) -> list[list[str]]:
        """
        genera los subgrafos conexos.
        !!! - Asume que el subgrafo es de tipo camino, pero no lo comprueba! para eso están los tests
        """
        if len(self.line_ids) < order:
            return []
        if (max_subgraphs_to_generate is not None) and (
            max_subgraphs_to_generate < (len(self.line_ids) - order + 1)
        ):
            random_sequence = np.random.choice(
                range(len(self.lines) - order + 1), size=max_subgraphs_to_generate
            )
            return [self.line_ids[i : i + order] for i in random_sequence]
        else:
            return [
                self.line_ids[i : i + order] for i in range(len(self.lines) - order + 1)
            ]

    def _calculate_total_area_and_centroid(self):
        self.centroid: np.ndarray = np.zeros((2,))
        self.total_words: int = 0
        total_area = 0

        for line in self.lines:
            self.total_words += len(line.text.split())
            area = line.polygon.area

            self.centroid += np.array(line.centroid()) * area
            total_area += area

        assert self.total_words > 0, "Se ha pasado un párrafo sin palabras."

        self.centroid /= total_area

        self.avg_rotation = self._get_average_rotation(
            [line.rotation for line in self.lines],
            [line.polygon.area for line in self.lines],
        )

        self.top: float = min([line.top for line in self.lines])
        self.left: float = min([line.left for line in self.lines])
        self.bot = max([line.bot for line in self.lines])
        self.right = max([line.right for line in self.lines])

    def _sort_lines_using_centroid_and_subgraph(self) -> None:

        if self.subgraph is None:
            raise ValueError("Cannot sort lines for a paragraph with a null subgraph.")
        theta_rad = -np.radians(-self.avg_rotation)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        cx_para, cy_para = self.centroid

        for line in self.lines:
            cx, cy = line.centroid()

            dx = cx - cx_para
            dy = cy - cy_para

            corrected_x = dx * cos_theta - dy * sin_theta + cx_para
            corrected_y = dx * sin_theta + dy * cos_theta + cy_para

            line.corrected_centroid = (
                corrected_x,
                corrected_y,
            )

        if not is_path_graph(
            self.subgraph
        ):  # si no es un grafo camino, empleamos el orden de lectura dado por las proyecciones
            self.lines = sorted(
                self.lines,
                key=lambda line: (
                    line.corrected_centroid[1],
                    line.corrected_centroid[0],
                ),
            )
            return

        if len(self.lines) == 1:
            return

        terminal_vertices = [
            line
            for line in self.lines
            if len(self.subgraph[line.id]) == 1  # ty:ignore[not-subscriptable]
        ]
        assert len(terminal_vertices) == 2

        top_line = min(
            terminal_vertices,
            key=lambda line: (
                line.corrected_centroid[1],  # ty: ignore[not-subscriptable]
                line.corrected_centroid[0],  # ty: ignore[not-subscriptable]
            ),
        )

        lines_by_id = {line.id: line for line in self.lines}
        ordered_lines = [top_line]
        visited = {top_line.id}
        previous_id: str | None = None
        current_id = top_line.id

        while len(ordered_lines) < len(self.lines):
            next_candidates = [
                neighbor_id
                for neighbor_id in self.subgraph[
                    current_id
                ]  # ty:ignore[not-subscriptable]
                if neighbor_id != previous_id and neighbor_id not in visited
            ]
            assert len(next_candidates) == 1

            next_id = next_candidates[0]
            ordered_lines.append(lines_by_id[next_id])
            visited.add(next_id)
            previous_id, current_id = current_id, next_id

        self.lines = ordered_lines
