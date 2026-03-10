from preprocessing.TextFragment import TextFragment
from preprocessing.ImageBox import ImageBox
from preprocessing.helpers.helper_to_classes import compose_collage, unrotate_image
from shapely import coverage_union_all
from shapely.affinity import affine_transform
import numpy as np
from PIL import Image


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
    )

    def __init__(
        self,
        image_boxes: list[ImageBox] | None = None,
        text_fragments: list[TextFragment] | None = None,
        task_id: int | None = None,
        index: int | None = None,
    ):
        assert (
            image_boxes or text_fragments
        ), "O bien image_boxes o bien text_fragments debe ser una lista no vacía"

        if image_boxes and text_fragments:
            assert set([box.fragment for box in image_boxes]) == set(
                [fragment.box for fragment in text_fragments]
            ), "Si se dan tanto image_boxes como text_fragments, deben corresponderse entre ellos."

        if not image_boxes:
            image_boxes = [f.box for f in text_fragments]
        elif not text_fragments:
            text_fragments = [b.fragment for b in image_boxes]

        self.image_boxes: list[ImageBox] = image_boxes
        self.text_fragments: list[TextFragment] = text_fragments
        self.task_id = task_id
        self.index = index

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

        self.image_boxes = sorted(
            self.image_boxes,
            key=lambda box: tuple(box.corrected_centroid[::-1]),
        )

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

    def collage(self, fill_color: tuple[int] = (255, 0, 255)):
        return compose_collage(self.image_boxes, fill_color)

    def transcription(self):
        return " ".join([fragment.text for fragment in self.text_fragments])

    def cluster_reading_order(
        self, unrotate: bool = False, fill_color: tuple[int] | None = (255, 0, 255)
    ):
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
        return coverage_union_all([box.polygon for box in self.image_boxes])

    def corrected_polygon(self, box: ImageBox):
        # TODO: check this works as expected
        t = np.radians(self.avg_rotation)
        a = np.cos(t)
        b = -np.sin(t)
        c = np.sin(t)
        d = np.cos(t)
        x_c = -self.centroid[0]
        y_c = -self.centroid[1]
        return affine_transform(box.polygon, [a, b, c, d, -x_c, -y_c])

    @staticmethod
    def _get_average_rotation(angles_in_degrees: list[float], areas: list[float]):

        angles_in_radians = np.radians(angles_in_degrees)
        sum_sin = np.sum(np.sin(np.radians(angles_in_radians)) * np.array(areas))
        sum_cos = np.sum(np.cos(np.radians(angles_in_radians)) * np.array(areas))
        return -np.degrees(np.arctan2(sum_sin, sum_cos))
