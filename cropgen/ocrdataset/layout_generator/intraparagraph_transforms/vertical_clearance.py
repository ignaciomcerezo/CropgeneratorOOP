from cropgen.processing import Paragraph
from cropgen.ocrdataset.layout_generator import (
    IntraparagraphTransform,
    _ParagraphInfo,
)
from shapely.affinity import translate
import numpy as np


class VerticalClearance(IntraparagraphTransform):
    def __init__(
        self,
        relative: float | None = None,
        absolute: float | None = None,
        modulate_by_probability: bool = False,
    ):
        if (absolute is not None) and (relative is not None):
            raise ValueError("Only one of 'absolute' or 'relative' can be provided.")
        elif (absolute is None) and (relative is None):
            raise ValueError("One of 'absolute' or 'relative' must be provided.")

        self.__absolute = absolute
        self.__relative = relative
        self.modulate_by_probability = modulate_by_probability

    def __call__(self, paragraph: Paragraph) -> None:
        info = _ParagraphInfo(paragraph)
        vertical_size = abs(
            info.box_bounds[0][1] - info.box_bounds[-1][1]
        )  # topmost's topmost to botmost's topmost

        if len(paragraph) < 2:
            print("Paragraph too small.")
            return

        Delta: float = (
            self.__absolute
            if self.__absolute is not None
            else self.__relative * vertical_size  # ty:ignore[unsupported-operator]
        )  # total difference in size

        delta_i = Delta / (len(paragraph) - 1)

        for k, box in enumerate(paragraph.image_boxes, start=1):
            # -Delta moves upwards, as topmost vertex has the most negative y coordinate

            if k:
                perturbation = (
                    1 if not self.modulate_by_probability else np.random.rand()
                )
                displacement = -Delta / 2 + delta_i * (k + perturbation - 2)
            else:
                displacement = -Delta / 2

            box.polygon = translate(box.polygon, yoff=displacement)
