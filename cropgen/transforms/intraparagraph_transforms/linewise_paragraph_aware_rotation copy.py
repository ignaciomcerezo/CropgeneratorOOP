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
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[Polygon]]:
        new_imgs = []
        new_polys = []
        for line in line_group:

            orig_bounds = line.polygon.bounds

            x0, y0, x1, y1 = orig_bounds
            center = (
                (x0 + x1) / 2,
                (y0 + y1) / 2,
            )

            new_poly = LinewiseParagraphAwareRotation().rotate_poly(
                line.polygon,
                line.rotation,
                center,
            )

            new_imgs.append(
                LinewiseParagraphAwareRotation.rotate_img(
                    line.stroke_crop,
                    line.rotation,
                    orig_bounds,
                    new_poly.bounds,
                )
            )
            new_polys.append(new_poly)
        return new_imgs, new_polys
