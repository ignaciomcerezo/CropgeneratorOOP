from cropgen.shared.parameters import Parameter
from typing import Sequence
from cropgen.transforms.transforms import IntraparagraphTransform
from cropgen.processing.paragraph import Paragraph
from cropgen.processing.line import Line
from shapely.geometry import Polygon
import shapely
import numpy as np
import cv2
from PIL import Image


class VerticalArch(IntraparagraphTransform):
    """
    Applies a parabolic arch to all lines.
    """

    def __init__(
        self, amplitude: Parameter | float, *, segmentation_thinness: int = 200
    ):
        self.amplitude = Parameter(amplitude)
        self.segmentation_thinness = segmentation_thinness

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

        x0 = min(polygon.bounds[0] for polygon in polygons)
        xf = max(polygon.bounds[2] for polygon in polygons)

        new_polygons = []
        new_images = []

        for image, polygon in zip(images, polygons):
            orig_bounds = polygon.bounds

            amplitude = self.amplitude()

            new_polygon = self._apply_arch_poly(polygon, amplitude, x0, xf)

            new_images.append(
                self._apply_arch_img(
                    image,
                    amplitude,
                    x0,
                    xf,
                    orig_bounds,
                    new_polygon.bounds,
                )
            )
            new_polygons.append(new_polygon)

        return new_images, new_polygons

    def _apply_arch_img(
        self,
        pil_img: Image.Image,
        amplitude: float,
        x0: float,
        xf: float,
        orig_bounds: tuple,
        new_bounds: tuple,
    ) -> Image.Image:
        """
        Applies an arch transformation strictly bounded by the polygon's new spatial footprint.
        """
        img_array = np.array(pil_img)

        orig_box_x0, orig_box_y0, _, _ = orig_bounds
        new_box_x0, new_box_y0, new_box_x2, new_box_y2 = new_bounds

        new_width = max(1, int(np.ceil(new_box_x2 - new_box_x0)))
        new_height = max(1, int(np.ceil(new_box_y2 - new_box_y0)))

        x_d, y_d = np.meshgrid(np.arange(new_width), np.arange(new_height))

        x_global = new_box_x0 + x_d

        domain_width = xf - x0
        if domain_width == 0:
            x_norm = np.zeros_like(x_global, dtype=np.float32)
        else:
            x_norm = (2.0 * x_global - (xf + x0)) / domain_width

        abs_amp = abs(amplitude)
        direction = "up" if amplitude < 0 else "down"

        if direction == "up":
            displacement = abs_amp * (x_norm**2)
        else:
            displacement = abs_amp * (1.0 - x_norm**2)

        x_s = (x_d + (new_box_x0 - orig_box_x0)).astype(np.float32)
        y_s = (y_d + (new_box_y0 - orig_box_y0) - displacement).astype(np.float32)

        arched_array = cv2.remap(
            img_array,
            x_s,
            y_s,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return Image.fromarray(arched_array)

    def _apply_arch_poly(
        self,
        poly: Polygon,
        amplitude: float,
        x0: float,
        xf: float,
    ) -> Polygon:
        """
        Mirrors the arch transformation using vectorized Shapely operations.
        """
        densified_poly = shapely.segmentize(poly, self.segmentation_thinness)
        abs_amp = abs(amplitude)
        direction = "up" if amplitude < 0 else "down"
        domain_width = xf - x0

        def vectorized_mapping(coords):
            out = np.empty_like(coords, dtype=np.float64)
            x = coords[:, 0]
            y = coords[:, 1]

            if domain_width == 0:
                x_norm = np.zeros_like(x)
            else:
                x_norm = (2.0 * x - (xf + x0)) / domain_width

            if direction == "up":
                displacement = abs_amp * (x_norm**2)
            else:
                displacement = abs_amp * (1.0 - x_norm**2)

            out[:, 0] = x
            out[:, 1] = y + displacement

            if coords.shape[1] == 3:
                out[:, 2] = coords[:, 2]

            return out

        return shapely.transform(densified_poly, vectorized_mapping)
