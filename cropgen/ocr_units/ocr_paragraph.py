from cropgen.ocr_units.ocr_line import OCRLine
from typing import Optional, Iterator

import numpy as np
from shapely import unary_union
from shapely import Polygon
from shapely.affinity import affine_transform

from cropgen.ocr_units.helpers.helper_to_classes import (
    is_path_graph,
)


class OCRParagraph:
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
        "_index",
    )

    def __init__(
        self,
        *,
        lines: list[OCRLine],
        task_id: int,
        subgraph: dict[str, set[str]],
        index: int | None = None,
    ):

        if not lines:
            raise ValueError("Lines cannot be empty.")

        if len(subgraph) != len(lines):
            raise ValueError(
                "The length of the subgraph passed to an OCRParagraph must be equal to the number of lines it contains."
            )

            r

        self.lines = lines
        self.task_id: int | None = task_id
        self._index: int | None = index

        self._calculate_total_area_and_centroid()

        self._sort_lines_using_centroid_and_subgraph(subgraph)

        self.line_ids = [line.id for line in self.lines]

        if set(subgraph.keys()) != set(self.line_ids):
            raise ValueError("The subgraph keys does not match the lines passed.")

        for line in self.lines:
            if line.paragraph_index is not None:
                raise ValueError("Line with paragraph already set.")
            line.paragraph_index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value: int):
        self._index = value
        for line in self.lines:
            line.paragraph_index = value

    def __iter__(self) -> Iterator[OCRLine]:
        for x in self.lines:
            yield x

    def __getitem__(self, index) -> OCRLine:
        return self.lines[index]

    def __lt__(self, other: "OCRParagraph"):
        return (self.top, self.left) < (other.top, other.left)

    def __gt__(self, other: "OCRParagraph"):
        return (self.top, self.left) > (other.top, other.left)

    def transcription(self, separator: str = " "):
        return separator.join([fragment.text for fragment in self.lines])

    def __len__(self):
        return len(self.line_ids)

    def __repr__(self):
        return f"<{self.index}-th paragraph of order {len(self)} contained in AnnotatedPage of task ({self.task_id})>"

    @staticmethod
    def _get_average_rotation(
        angles_in_degrees: list[float], areas: list[float]
    ) -> float:

        angles_in_radians = np.radians(angles_in_degrees)
        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))
        return -float(np.degrees(np.arctan2(sum_sin, sum_cos)))

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

    def _sort_lines_using_centroid_and_subgraph(
        self, subgraph: dict[str, set[str]]
    ) -> None:

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
            subgraph
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

        terminal_vertices = [line for line in self.lines if len(subgraph[line.id]) == 1]
        assert len(terminal_vertices) == 2

        top_line = min(
            terminal_vertices,
            key=lambda line: (
                line.corrected_centroid[1],
                line.corrected_centroid[0],
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
                for neighbor_id in subgraph[current_id]
                if neighbor_id != previous_id and neighbor_id not in visited
            ]
            assert len(next_candidates) == 1

            next_id = next_candidates[0]
            ordered_lines.append(lines_by_id[next_id])
            visited.add(next_id)
            previous_id, current_id = current_id, next_id

        self.lines = ordered_lines
