from typing import Callable, Sequence
import numpy as np
from PIL import Image
from PIL.ImageTransform import PerspectiveTransform
from shapely.geometry import Polygon
from shapely.ops import transform

from cropgen.processing import Line, Paragraph
from cropgen.shared.parameters import Parameter
from cropgen.transforms.transforms import IntraparagraphTransform

point2D = tuple[float, float]


class PerspectiveTransformation(IntraparagraphTransform):
    def __init__(
        self,
        source_points: Sequence[point2D] = ((0, 0), (1, 0), (0, 1), (1, 1)),
        destination_points: Sequence[point2D] = ((0, 0), (1, 0), (0, 1), (1, 1)),
        *,
        noise_x: Parameter | float = 0,
        noise_y: Parameter | float = 0,
    ):
        self.source_points = [tuple(point) for point in source_points]
        self.destination_points = [tuple(point) for point in destination_points]
        self.noise_x = Parameter(noise_x)
        self.noise_y = Parameter(noise_y)

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

        noisy_destination_points = [
            (
                x + self.noise_x(),
                y + self.noise_y(),
            )
            for x, y in self.destination_points
        ]

        inv_coefficients = calculate_perspective_coefficients(
            self.source_points, noisy_destination_points
        )
        fwd_coefficients = calculate_perspective_coefficients(
            noisy_destination_points, self.source_points
        )

        shapely_configured_transform = _get_shapely_perspective_transform(
            *fwd_coefficients
        )
        image_configured_transform = PerspectiveTransform(inv_coefficients)

        for i, (image, polygon) in enumerate(zip(images, polygons)):
            polygons[i] = transform(shapely_configured_transform, polygon)

            original_width, original_height = image.size
            expected_size = _calculate_projected_size(
                original_width, original_height, fwd_coefficients
            )
            images[i] = image.transform(
                expected_size,
                image_configured_transform,
            )

        return images, polygons


def _calculate_projected_size(
    width: int, height: int, fwd_coeffs: tuple[float, ...]
) -> tuple[int, int]:
    """Projects image corners forward to determine the bounding line dimensions."""
    a, b, c, d, e, f, g, h = fwd_coeffs
    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    proj_x = []
    proj_y = []

    for x, y in corners:
        denominator = g * x + h * y + 1.0
        proj_x.append((a * x + b * y + c) / denominator)
        proj_y.append((d * x + e * y + f) / denominator)

    new_width = max(1, int(np.ceil(max(proj_x) - min(proj_x))))
    new_height = max(1, int(np.ceil(max(proj_y) - min(proj_y))))

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
