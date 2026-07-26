from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.layout_generator.layouts import (
    IntraparagraphTransform,
    _ParagraphInfo,
)
from shapely.affinity import translate


class add_vertical_clearance(IntraparagraphTransform):
    def __init__(
        self,
        absolute: float | None = None,
        relative: float | None = None,
    ):
        if (absolute is not None) and (relative is not None):
            raise ValueError("Only one of 'absolute' or 'relative' can be provided.")
        elif (absolute is None) and (relative is None):
            raise ValueError("One of 'absolute' or 'relative' must be provided.")

        self.__absolute = absolute
        self.__relative = relative

    def __call__(self, paragraph: Paragraph) -> None:
        info = _ParagraphInfo(paragraph)
        vertical_size = (
            info.bounds[0][1] - info.bounds[-1][1]
        )  # topmost's topmost to botmost's topmost

        if len(paragraph) < 2:
            return

        Delta: float = (
            self.__absolute
            if self.__absolute is not None
            else self.__relative * vertical_size  # ty:ignore[unsupported-operator]
        )  # total difference in size

        delta_i = Delta / (len(paragraph) - 1)

        for k, box in enumerate(paragraph.image_boxes, start=1):
            # -Delta moves upwards, as topmost vertex has the most negative y coordinate
            displacement = -Delta / 2 + delta_i * (k - 1)

            box.polygon = translate(box.polygon, yoff=displacement)
