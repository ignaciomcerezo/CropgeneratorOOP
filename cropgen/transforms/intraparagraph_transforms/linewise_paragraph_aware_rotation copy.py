from cropgen.processing.helpers.helper_to_classes import calculate_reading_angle
from cropgen.shared.parameters import Parameter
from cropgen.processing.line import Line
from cropgen.processing import Paragraph
from typing import Optional, Literal, Sequence

from cropgen.transforms.transforms import (
    IntraparagraphTransform,
)
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from cropgen.transforms.intraparagraph_transforms.linewise_paragraph_aware_rotation import (
    LinewiseParagraphAwareRotation,
)

import numpy as np
import cv2
import shapely
from shapely import Polygon
from shapely.affinity import rotate

from PIL import Image


class StraightenLines(IntraparagraphTransform):
    """
    Rotates the lines of a paragraph, leaving them at 0 rotation each.
    """

    def __init__(self):
        return

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

        for i, (image, polygon) in enumerate(zip(images, polygons)):

            orig_bounds = polygon.bounds
            rotation = calculate_reading_angle(polygon)

            x0, y0, x1, y1 = orig_bounds
            center = (
                (x0 + x1) / 2,
                (y0 + y1) / 2,
            )

            polygons[i] = LinewiseParagraphAwareRotation().rotate_poly(
                polygon,
                rotation,  # TODO: does this go with a - sign? - using LayoutGenerator and .manuscript with polygons should be obvious
                center,
            )

            images[i] = LinewiseParagraphAwareRotation.rotate_img(
                image,
                rotation,  # TODO: does THIS? ¿?¿ - using LayoutGenerator and .manuscript with polygons should be obvious
                orig_bounds,
                polygons[i].bounds,
            )

        return images, polygons
