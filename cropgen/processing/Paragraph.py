from typing import Optional

import numpy as np
from PIL import Image
from shapely import coverage_union_all
from shapely.affinity import affine_transform

from cropgen.processing.ImageBox import ImageBox
from cropgen.processing.TextFragment import TextFragment
from cropgen.processing.helpers.helper_to_classes import (
    compose_collage,
    unrotate_image,
    is_path_graph,
)


class Paragraph:
    __slots__ = (
        "image_boxes",
        "text_fragments",
        "centroid",
        "total_words",
        "avg_rotation",
        "top",
        "left",
        "image_boxes_ids",
        "text_fragments_ids",
        "task_id",
        "index",
        "subgraph",
    )

    def __init__(
        self,
        image_boxes: list[ImageBox] | None = None,
        text_fragments: list[TextFragment] | None = None,
        task_id: int | None = None,
        index: int | None = None,
        subgraph: dict[str, set[str]] | None = None,
    ):
        assert (
            image_boxes or text_fragments
        ), "O bien image_boxes o bien text_fragments debe ser una lista no vacía"

        if image_boxes and text_fragments:
            assert set([box.fragment for box in image_boxes]) == set(
                [fragment.box for fragment in text_fragments]
            ), "Si se dan tanto image_boxes como text_fragments, deben corresponderse entre ellos."

        if not image_boxes:
            assert isinstance(text_fragments, list)
            image_boxes: list[ImageBox] = [f.box for f in text_fragments]
        elif not text_fragments:
            assert isinstance(image_boxes, list)
            text_fragments: list[TextFragment] = [b.fragment for b in image_boxes]

        self.image_boxes: list[ImageBox] = image_boxes
        self.text_fragments: list[TextFragment] = text_fragments
        self.task_id: int | None = task_id
        self.index: int | None = index
        self.subgraph: Optional[dict[str, set[str]]] = subgraph

        self._calcualate_total_area_and_centroid()

        self._sort_image_boxes_using_centroid_and_subgraph()

        # reordenamos
        self.text_fragments = [box.fragment for box in self.image_boxes]

        self.image_boxes_ids = [box.id for box in self.image_boxes]
        self.text_fragments_ids = [fragment.id for fragment in self.text_fragments]

    def __lt__(
        self, other: "Paragraph"
    ):  # para poder ordenar automáticamente usando list.sort o sorted()
        return (self.top, self.left) < (other.top, other.left)

    def __gt__(self, other: "Paragraph"):
        return (self.top, self.left) > (other.top, other.left)

    def collage(
        self,
        fill_color: tuple[int, int, int] | tuple[int, int, int, int] = (255, 0, 255),
    ):
        return compose_collage(self.image_boxes, fill_color)

    def transcription(self, separator: str = " "):
        return separator.join([fragment.text for fragment in self.text_fragments])

    def cluster_reading_order(
        self,
        unrotate: bool = False,
        fill_color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    ):
        fill_color: tuple[int, int, int] | tuple[int, int, int, int] = (
            tuple(fill_color) if fill_color is not None else (255, 0, 255)
        )

        if not unrotate:
            collage = compose_collage(self.image_boxes, (255, 0, 255))
        else:
            transp_collage = compose_collage(
                self.image_boxes, tuple(list(fill_color) + [0])
            )

            unrotated = unrotate_image(transp_collage, -self.avg_rotation)

            collage = Image.new("RGB", unrotated.size, fill_color)
            collage.paste(unrotated, (0, 0), mask=unrotated)

        return (
            collage,
            " ".join([fragment.text for fragment in self.text_fragments]),
            self.text_fragments[0].starting_index,
        )

    def __len__(self):
        return len(self.image_boxes_ids)

    def __repr__(self):
        return f"<{self.index}-th paragraph of order {len(self)} contained in AnnotatedPage of task ({self.task_id})>"

    def union_polygon(self):
        print("¡Recuerda que la y está invertida!")
        return coverage_union_all([box.polygon for box in self.image_boxes])

    def corrected_polygon(self, box: ImageBox):
        t: float = np.radians(self.avg_rotation)
        a: float = np.cos(t)
        b: float = -np.sin(t)
        c: float = np.sin(t)
        d: float = np.cos(t)
        x_c: float = -float(self.centroid[0])
        y_c: float = -float(self.centroid[1])
        return affine_transform(box.polygon, [a, b, c, d, -x_c, -y_c])

    @staticmethod
    def _get_average_rotation(
        angles_in_degrees: list[float], areas: list[float]
    ) -> float:

        angles_in_radians = np.radians(angles_in_degrees)
        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))
        return -float(np.degrees(np.arctan2(sum_sin, sum_cos)))

    def generate_conntected_subgraphs(
        self, order: int, max_subgraphs_to_generate: Optional[int] = None
    ) -> list[list[str]]:
        """
        genera los subgrafos conexos.
        !!! - Asume que el subgrafo es de tipo camino, pero no lo comprueba! para eso están los tests
        """
        if len(self.image_boxes_ids) < order:
            return []
        if (max_subgraphs_to_generate is not None) and (
            max_subgraphs_to_generate < (len(self.image_boxes_ids) - order + 1)
        ):
            random_sequence = np.random.choice(
                range(len(self.image_boxes) - order + 1), size=max_subgraphs_to_generate
            )
            return [self.image_boxes_ids[i : i + order] for i in random_sequence]
        else:
            return [
                self.image_boxes_ids[i : i + order]
                for i in range(len(self.image_boxes) - order + 1)
            ]

    def _calcualate_total_area_and_centroid(self):
        self.centroid: np.ndarray = np.zeros((2,))
        self.total_words: int = 0
        total_area = 0

        for image_box in self.image_boxes:
            self.total_words += len(image_box.fragment.text.split())
            area = image_box.polygon.area

            self.centroid += np.array(image_box.centroid()) * area
            total_area += area

        assert self.total_words > 0, "Se ha pasado un párrafo sin palabras."

        self.centroid /= total_area

        self.avg_rotation = self._get_average_rotation(
            [box.rotation for box in self.image_boxes],
            [box.polygon.area for box in self.image_boxes],
        )

        self.top: float = min([box.top for box in self.image_boxes])
        self.left: float = min([box.left for box in self.image_boxes])

    def _subgraph_is_Pk(self) -> bool:
        return self.subgraph is not None and is_path_graph(self.subgraph)

    def _sort_image_boxes_using_centroid_and_subgraph(self):
        theta_rad = -np.radians(-self.avg_rotation)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        cx_para, cy_para = self.centroid

        for image_box in self.image_boxes:
            cx, cy = image_box.centroid()

            dx = cx - cx_para
            dy = cy - cy_para

            corrected_x = dx * cos_theta - dy * sin_theta + cx_para
            corrected_y = dx * sin_theta + dy * cos_theta + cy_para

            image_box.corrected_centroid = (corrected_x, corrected_y)

        if (
            not self._subgraph_is_Pk()
        ):  # si no es un grafo camino, empleamos el orden de lectura dado por las proyecciones
            self.image_boxes = sorted(
                self.image_boxes,
                key=lambda box: (box.corrected_centroid[1], box.corrected_centroid[0]),  # ty:ignore[not-subscriptable]
            )
            return self.image_boxes

        if len(self.image_boxes) == 1:
            return self.image_boxes

        terminal_vertices = [
            box for box in self.image_boxes if len(self.subgraph[box.id]) == 1  # ty:ignore[not-subscriptable]
        ]
        assert len(terminal_vertices) == 2

        top_box = min(
            terminal_vertices,
            key=lambda box: (box.corrected_centroid[1], box.corrected_centroid[0]),  # ty:ignore[not-subscriptable]
        )

        boxes_by_id = {box.id: box for box in self.image_boxes}
        ordered_boxes = [top_box]
        visited = {top_box.id}
        previous_id: str | None = None
        current_id = top_box.id

        while len(ordered_boxes) < len(self.image_boxes):
            next_candidates = [
                neighbor_id
                for neighbor_id in self.subgraph[current_id]  # ty:ignore[not-subscriptable]
                if neighbor_id != previous_id and neighbor_id not in visited
            ]
            assert len(next_candidates) == 1

            next_id = next_candidates[0]
            ordered_boxes.append(boxes_by_id[next_id])
            visited.add(next_id)
            previous_id, current_id = current_id, next_id

        self.image_boxes = ordered_boxes
        return self.image_boxes
