from cropgen.processing.line import Line
from cropgen.processing import Paragraph
from typing import Optional, Literal, Sequence

from cropgen.transforms.transforms import (
    IntraparagraphTransform,
    ParagraphInfo,
)

import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon
from shapely.affinity import rotate

from PIL import Image


class LinewiseRotation(IntraparagraphTransform):
    """
    Rotates the lines of a paragraph individually.
    """

    def __init__(
        self,
        *,
        relative: Optional[float] = None,
        absolute: Optional[float] = None,
        metric: Literal[
            "degrees",
            "pi radians",
            "radians",
        ] = "degrees",
    ):
        if relative is None and absolute is None:
            raise ValueError("Either relative or absolute rotations must be provided")

        self._relative = relative
        self._absolute = absolute
        self._metric = metric

    def __call__(
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[Polygon]]:

        info = ParagraphInfo(line_group)

        if self._relative is not None:
            rotation = info.avg_rotation * self._relative

        else:
            if self._absolute is None:
                raise ValueError("Either relative or absolute must be provided")

            match self._metric:
                case "degrees":
                    rotation = self._absolute

                case "radians":
                    rotation = self._absolute / np.pi * 180

                case "pi radians":
                    rotation = self._absolute * 180

                case _:
                    raise ValueError(f"Unknown metric: {self._metric}")

        new_images = []
        new_polygons = []

        for line in line_group:

            # The polygon is in global/page coordinates.
            orig_bounds = line.polygon.bounds

            # Rotate around the center of the LINE, not the local
            # pixel coordinates of the crop.
            x0, y0, x1, y1 = orig_bounds
            center = (
                (x0 + x1) / 2,
                (y0 + y1) / 2,
            )

            new_poly = self._rotate_poly(
                line.polygon,
                rotation,
                center,
            )

            new_images.append(
                self._rotate_img(
                    line.stroke_crop,
                    rotation,
                    orig_bounds,
                    new_poly.bounds,
                )
            )
            new_polygons.append(new_poly)

        return new_images, new_polygons

    @staticmethod
    def _rotate_poly(
        poly: Polygon,
        angle: float,
        center: tuple[float, float],
    ) -> Polygon:

        return rotate(
            poly,
            angle,
            origin=center,
            use_radians=False,
        )

    @staticmethod
    def _rotate_img(
        pil_img: Image.Image,
        angle: float,
        orig_bounds: tuple[float, float, float, float],
        new_bounds: tuple[float, float, float, float],
    ) -> Image.Image:

        img_array = np.asarray(pil_img)

        orig_x0, orig_y0, _, _ = orig_bounds
        new_x0, new_y0, new_x1, new_y1 = new_bounds

        new_width = max(
            1,
            int(np.ceil(new_x1 - new_x0)),
        )

        new_height = max(
            1,
            int(np.ceil(new_y1 - new_y0)),
        )

        x_d, y_d = np.meshgrid(
            np.arange(new_width),
            np.arange(new_height),
        )

        x_global = new_x0 + x_d
        y_global = new_y0 + y_d

        orig_x1 = orig_bounds[2]
        orig_y1 = orig_bounds[3]

        cx = (orig_x0 + orig_x1) / 2
        cy = (orig_y0 + orig_y1) / 2

        theta = np.radians(angle)

        c = np.cos(theta)
        s = np.sin(theta)

        x_source_global = c * (x_global - cx) + s * (y_global - cy) + cx

        y_source_global = -s * (x_global - cx) + c * (y_global - cy) + cy

        x_source = (x_source_global - orig_x0).astype(np.float32)

        y_source = (y_source_global - orig_y0).astype(np.float32)

        rotated_array = cv2.remap(
            img_array,
            x_source,
            y_source,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return Image.fromarray(rotated_array)
