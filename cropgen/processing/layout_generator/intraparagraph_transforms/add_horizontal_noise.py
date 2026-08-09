from dataclasses import dataclass
from typing import Literal, Callable, Self


from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.layout_generator.transforms import (
    IntraparagraphTransform,
    _ParagraphInfo,
)
from shapely.affinity import translate
import numpy as np

NOISES = Literal["linear", "wave", "random", "zigzag"]


@dataclass
class AddHorizontalNoise(IntraparagraphTransform):
    noise_type: NOISES
    period: float = 10
    amplitude: float = 5
    slope: float = 5
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
        x0, y0 = paragraph.image_boxes[0].centroid()
        for box in paragraph.image_boxes:
            xi, yi = box.centroid()
            box.polygon = translate(
                box.polygon, xoff=self.amplitude * np.cos(2 * np.pi * 1 / self.period)
            )

    def _call_zigzag(self, paragraph: Paragraph) -> None:
        for i, box in enumerate(paragraph.image_boxes):
            box.polygon = translate(
                box.polygon, xoff=(1 - 2 * (i % 2)) * self.amplitude
            )

    def _call_random(self, paragraph: Paragraph) -> None:
        for box in paragraph.image_boxes:
            box.polygon = translate(
                box.polygon, xoff=(np.random.rand() * 2 - 1) * self.amplitude
            )

    def __call__(self, paragraph: Paragraph) -> None:
        self.type2map[self.noise_type](paragraph)
