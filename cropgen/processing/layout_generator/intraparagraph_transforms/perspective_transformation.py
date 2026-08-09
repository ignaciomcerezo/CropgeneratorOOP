from shapely.ops import transform
from typing import Callable
from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.layout_generator.transforms import IntraparagraphTransform
from PIL.ImageTransform import PerspectiveTransform
import numpy as np

point2D = tuple[float, float]


class PerspectiveTransformation(IntraparagraphTransform):
    def __init__(self, source_points: list, destination_points: list):

        # PIL needs the coeff. of the inverse transform, while shapely needs those of the direct one
        self.inv_coefficients = calculate_perspective_coefficients(
            source_points, destination_points
        )
        self.fwd_coefficients = calculate_perspective_coefficients(
            destination_points, source_points
        )

    def __call__(self, paragraph):
        shapely_configured_transform = _get_shapely_perspective_transform(
            *self.fwd_coefficients
        )

        image_configured_transform = PerspectiveTransform(self.inv_coefficients)

        for box in paragraph.image_boxes:
            box.polygon = transform(shapely_configured_transform, box.polygon)

            original_width, original_height = box.crop.size
            expected_size = _calculate_projected_size(
                original_width, original_height, self.fwd_coefficients
            )

            box.crop = box.crop.transform(expected_size, image_configured_transform)


def _calculate_projected_size(
    width: int, height: int, fwd_coeffs: tuple[float, ...]
) -> tuple[int, int]:
    """Projects image corners forward to determine the bounding box dimensions."""
    a, b, c, d, e, f, g, h = fwd_coeffs
    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    proj_x = []
    proj_y = []

    for x, y in corners:
        denominator = g * x + h * y + 1.0
        proj_x.append((a * x + b * y + c) / denominator)
        proj_y.append((d * x + e * y + f) / denominator)

    new_width = int(np.ceil(max(proj_x) - min(proj_x)))
    new_height = int(np.ceil(max(proj_y) - min(proj_y)))

    return new_width, new_height


def calculate_perspective_coefficients(
    source_points: list, destination_points: list
) -> tuple[float, ...]:
    A = []
    b = []
    for (u, v), (x, y) in zip(source_points, destination_points):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b.extend([u, v])

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    coeffs = np.linalg.solve(A, b)
    return tuple(coeffs)


def _get_shapely_perspective_transform(
    *coefficients: float,
) -> Callable[[float, float], tuple[float, float]]:
    a, b, c, d, e, f, g, h = coefficients

    def _configured_perspective_transform(x, y):
        denominator = g * x + h * y + 1.0
        x_proj = (a * x + b * y + c) / denominator
        y_proj = (d * x + e * y + f) / denominator
        return x_proj, y_proj

    return _configured_perspective_transform
