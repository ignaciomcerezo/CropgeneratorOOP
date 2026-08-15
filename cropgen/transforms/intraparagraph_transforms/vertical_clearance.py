from cropgen.shared.parameters import Parameter
from typing import Sequence
from shapely.geometry import Polygon
from PIL import Image
from cropgen.processing import Paragraph, Line
from cropgen.transforms.transforms import (
    IntraparagraphTransform,
)
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from shapely.affinity import translate
import numpy as np


class VerticalClearance(IntraparagraphTransform):
    def __init__(
        self,
        relative: Parameter | float | None = None,
        absolute: Parameter | float | None = None,
        add_probabilistic_noise: bool = False,
    ):
        if (absolute is not None) and (relative is not None):
            raise ValueError("Only one of 'absolute' or 'relative' can be provided.")
        elif (absolute is None) and (relative is None):
            raise ValueError("One of 'absolute' or 'relative' must be provided.")

        self.__absolute = Parameter(absolute) if absolute is not None else None
        self.__relative = Parameter(relative) if relative is not None else None
        self.noise = add_probabilistic_noise

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph
            | Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)

        info = LineGroupInfo.from_polygons(polygons)
        vertical_size = abs(
            info.box_bounds[0][1] - info.box_bounds[-1][1]
        )  # topmost's topmost to botmost's topmost

        if len(polygons) < 2:
            raise ValueError("Paragraph too small")

        Delta: float = (
            self.__absolute()
            if self.__absolute is not None
            else self.__relative() * vertical_size  # ty: ignore[call-non-callable]
        )  # total difference in size

        delta_i = Delta / (len(polygons) - 1)

        new_images = []
        new_polygons = []

        for k, (image, polygon) in enumerate(zip(images, polygons), start=1):
            # -Delta moves upwards, as topmost vertex has the most negative y coordinate

            if k:
                perturbation = 1 if not self.noise else np.random.rand()
                displacement = -Delta / 2 + delta_i * (k + perturbation - 2)
            else:
                displacement = -Delta / 2

            new_polygons.append(translate(polygon, yoff=displacement))
            new_images.append(image)

        return new_images, new_polygons
