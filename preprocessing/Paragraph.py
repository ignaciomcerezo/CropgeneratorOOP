from preprocessing.classes.TextFragment import TextFragment
from preprocessing.classes.ImageBox import ImageBox
import numpy as np


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
    )

    def __init__(
        self,
        image_boxes: list[ImageBox] | None = None,
        text_fragments: list[TextFragment] | None = None,
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

        self.centroid: np.ndarray = np.zeros((2,))
        self.total_words: int = 0
        self.avg_rotation: float = 0

        for image_box in self.image_boxes:
            words = len(image_box.fragment.text.split())
            self.centroid += np.array(image_box.centroid()) * words
            self.avg_rotation += image_box.rotation * words
            self.total_words += words
        assert self.total_words > 0, "Se ha pasado un párrafo sin palabras."

        self.centroid /= self.total_words
        self.avg_rotation /= self.total_words

        self.top: float = min([box.top for box in self.image_boxes])
        self.left: float = min([box.left for box in self.image_boxes])

        avg_rot_rad = np.radians(self.avg_rotation)

        for image_box in self.image_boxes:
            x_glob, y_glob = image_box.centroid()

            x = x_glob - self.centroid[0]
            y = y_glob - self.centroid[1]

            x_corr = (
                float(x * np.cos(avg_rot_rad) + y * np.sin(avg_rot_rad))
                + self.centroid[0]
            )
            y_corr = (
                float(-x * np.sin(avg_rot_rad) + y * np.cos(avg_rot_rad))
                + self.centroid[1]
            )

            image_box.corrected_centroid = (x_corr, y_corr)

        self.image_boxes = sorted(
            self.image_boxes, key=lambda box: box.corrected_centroid[::-1]
        )

        # reordenamos
        self.text_fragments = [box.fragment for box in self.image_boxes]

        self.image_boxes_ids = (box.id for box in self.image_boxes)
        self.text_fragments_ids = [fragment.id for fragment in self.text_fragments]

    def __lt__(
        self, other: "Paragraph"
    ):  # para poder ordenar automáticamente usando list.sort o sorted()
        return (self.top, self.left) < (other.top, other.left)

    def __gt__(self, other: "Paragraph"):
        return (self.top, self.left) > (other.top, other.left)
