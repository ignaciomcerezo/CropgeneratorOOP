from cropgen.shared.parameters import Parameter
from typing import Sequence
from cropgen.processing import Paragraph, Line
from shapely import Polygon
from shapely.affinity import rotate
import numpy as np
import cv2
from PIL import Image
from cropgen.transforms.transforms import (
    IntraparagraphTransform,
)
from cropgen.transforms.helpers.line_group_info import LineGroupInfo


class ParagraphwiseRotation(IntraparagraphTransform):
    """
    Rotates a whole paragraph around its centroid.
    """

    def __init__(
        self,
        *,
        relative: Parameter | float | None = None,
        absolute: Parameter | float | None = None,
        metric: str = "degrees",
    ):
        if relative is None and absolute is None:
            raise ValueError("Either relative or absolute rotations must be provided")

        self._relative = Parameter(relative) if relative is not None else None
        self._absolute = Parameter(absolute) if absolute is not None else None
        self._metric = metric

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

        if self._relative is not None:
            rotation = info.avg_rotation * self._relative()
        else:
            if self._absolute is None:
                raise ValueError("Either relative or absolute must be provided")

            match self._metric:
                case "degrees":
                    rotation = self._absolute()
                case "radians":
                    rotation = self._absolute() / np.pi * 180
                case "pi radians":
                    rotation = self._absolute() * 180
                case _:
                    raise ValueError(f"Unknown metric: {self._metric}")

        centroid = info.centroid

        for i, (image, polygon) in enumerate(zip(images, polygons)):

            orig_bounds = polygon.bounds

            polygons[i] = self._rotate_poly(
                polygon,
                rotation,
                centroid,
            )

            images[i] = self._rotate_img(
                image,
                rotation,
                centroid,
                orig_bounds,
                polygons[i].bounds,
            )

        return images, polygons

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
        center: tuple[float, float],
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

        cx, cy = center

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
