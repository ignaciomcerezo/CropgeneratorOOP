from cropgen.processing.Paragraph import Paragraph
from shapely.geometry import Polygon
from shapely import affinity
import shapely
import numpy as np
import cv2
from PIL import Image

from cropgen.ocrdataset.layout_generator.transforms import (
    IntraparagraphTransform,
    _ParagraphInfo,
)


class ParagraphwiseRotation(IntraparagraphTransform):
    """
    Rotates a whole paragraph around its centroid.
    """

    def __init__(
        self,
        *,
        relative: float | None = None,
        absolute: float | None = None,
        metric: str = "degrees",
    ):
        if relative is None and absolute is None:
            raise ValueError("Either relative or absolute rotations must be provided")

        self._relative = relative
        self._absolute = absolute
        self._metric = metric

    def __call__(self, paragraph: Paragraph):

        if self._relative is not None:
            rotation = paragraph.avg_rotation * self._relative
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

        centroid = _ParagraphInfo(paragraph).centroid

        for box in paragraph.image_boxes:

            orig_bounds = box.polygon.bounds

            new_poly = self._rotate_poly(
                box.polygon,
                rotation,
                centroid,
            )

            box.stroke_crop = self._rotate_img(
                box.stroke_crop,
                rotation,
                centroid,
                orig_bounds,
                new_poly.bounds,
            )

            box.polygon = new_poly

        return paragraph

    @staticmethod
    def _rotate_poly(
        poly: Polygon,
        angle: float,
        center: tuple[float, float],
    ) -> Polygon:

        return shapely.affinity.rotate(
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
