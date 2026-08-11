from dataclasses import dataclass
from typing import Literal, Callable


from cropgen.processing import Paragraph
from cropgen.ocrdataset.layout_generator.transforms import (
    IntraparagraphTransform,
    _ParagraphInfo,
)
from shapely.affinity import translate
import numpy as np

NOISES = Literal["linear", "wave", "random", "zigzag"]


@dataclass
class LinewiseHorizontalTranslation(IntraparagraphTransform):
    noise_type: NOISES
    period: float = 50
    amplitude: float = 5
    slope: float = 1
    intercept: float = 0

    def __post_init__(self):
        self.type2map: dict[NOISES, Callable[[Paragraph], None]] = {
            "linear": self._call_linear,
            "wave": self._call_wave,
            "random": self._call_random,
            "zigzag": self._call_zigzag,
        }

    def _call_linear(self, paragraph: Paragraph) -> None:
        x0, y0 = paragraph.image_boxes[0].centroid()
        for box in paragraph.image_boxes:
            xi, yi = box.centroid()
            box.polygon = translate(
                box.polygon, xoff=self.intercept + self.slope * abs(yi - y0)
            )

    def _call_wave(self, paragraph: Paragraph) -> None:
        _, y0 = paragraph.image_boxes[0].centroid()

        for box in paragraph.image_boxes:
            _, yi = box.centroid()

            xoff = self.amplitude * np.cos(2 * np.pi * (yi - y0) / self.period)

            box.polygon = translate(box.polygon, xoff=xoff)

    def _call_zigzag(self, paragraph: Paragraph) -> None:
        _, y0 = paragraph.image_boxes[0].centroid()

        for box in paragraph.image_boxes:
            _, yi = box.centroid()

            phase = int((yi - y0) / self.period)
            xoff = self.amplitude if phase % 2 == 0 else -self.amplitude

            box.polygon = translate(box.polygon, xoff=xoff)

    def _call_random(self, paragraph: Paragraph) -> None:
        for box in paragraph.image_boxes:
            xoff = np.random.uniform(-self.amplitude, self.amplitude)
            box.polygon = translate(box.polygon, xoff=xoff)

    def __call__(self, paragraph: Paragraph) -> None:
        self.type2map[self.noise_type](paragraph)
