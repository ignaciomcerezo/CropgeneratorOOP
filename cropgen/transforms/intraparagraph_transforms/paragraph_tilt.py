from typing import Callable, Sequence, Literal
import numpy as np
from PIL import Image
from PIL.ImageTransform import PerspectiveTransform
from shapely.geometry import Polygon
from shapely.ops import transform

from cropgen.processing import Line, Paragraph
from cropgen.shared.parameters import Parameter
from cropgen.transforms.helpers.line_group_info import LineGroupInfo, Vector2D
from cropgen.transforms.transforms import IntraparagraphTransform

point2D = tuple[float, float]


class ParagraphTilt(IntraparagraphTransform):
    def __init__(
        self,
        relative_strength: Parameter | float = 0.2,
        tilt_axis: Literal["vertical", "horizontal"] = "horizontal",
    ):
        self.relative = Parameter(relative_strength)
        self._tilt_horizontal = tilt_axis == "horizontal"

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph | Sequence[Line] | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)
        images = list(images)
        polygons = list(polygons)

        LGinfo = LineGroupInfo.from_polygons(polygons)
        mbr: Polygon = LineGroupInfo.polygon_union(polygons).minimum_rotated_rectangle
        reading_direction = LGinfo.reading_direction
        orthogonal_direction = LGinfo.orthogonal_direction

        is_right = lambda point: np.dot(orthogonal_direction, point - LGinfo.center) > 0
        is_up = lambda point: np.dot(reading_direction, point - LGinfo.center) > 0

        points: list[Vector2D] = [np.array(point) for point in mbr.exterior.coords[:-1]]
        a, b, c, d = None, None, None, None

        for point in points:
            if is_up(point):
                if is_right(point):
                    b = point
                else:
                    a = point
            else:
                if is_right(point):
                    c = point
                else:
                    d = point

        if any(obj is None for obj in (a, b, c, d)):
            raise ValueError("Geometry failed: polygons too mangled.")

        t = 0.5 * self.relative()
        if t <= -1 or t >= 1:
            raise ValueError("The strength of the tilt must lie in (-1, 1).")

        if not self._tilt_horizontal:
            a, b, c, d = a, d, b, c

        a_moved, b_moved = self._symm_lerp(a, b, t)  # ty: ignore[invalid-argument-type]
        c_moved, d_moved = self._symm_lerp(
            c, d, -t  # ty: ignore[invalid-argument-type]
        )

        source_points = [a, b, c, d]
        destination_points = [a_moved, b_moved, c_moved, d_moved]

        H_fwd = find_homography_matrix(
            source_points, destination_points  # ty: ignore[invalid-argument-type]
        )
        H_inv = np.linalg.inv(H_fwd)

        shapely_configured_transform = _get_shapely_perspective_transform(H_fwd)

        for i, (image, polygon) in enumerate(zip(images, polygons)):
            orig_bounds = polygon.bounds
            transformed_polygon = transform(shapely_configured_transform, polygon)
            polygons[i] = transformed_polygon
            trans_bounds = transformed_polygon.bounds

            expected_size, image_configured_transform = _get_pil_perspective_transform(
                H_inv, orig_bounds, trans_bounds
            )

            images[i] = image.transform(
                expected_size,
                image_configured_transform,
                resample=Image.Resampling.BICUBIC,
            )

        return images, polygons

    @staticmethod
    def _symm_lerp(A: np.ndarray, B: np.ndarray, t: float) -> tuple[Vector2D, Vector2D]:
        return ((1 - t) * A + t * B, t * A + (1 - t) * B)


def find_homography_matrix(
    source_points: Sequence[point2D | np.ndarray],
    destination_points: Sequence[point2D | np.ndarray],
) -> np.ndarray:
    """Computes the 3x3 forward projective matrix H mapping source -> destination."""
    A = []
    b = []
    for (u, v), (x, y) in zip(source_points, destination_points):
        A.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        A.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        b.extend([x, y])

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    h = np.linalg.solve(A, b)

    return np.array(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ],
        dtype=float,
    )


def _get_shapely_perspective_transform(
    H: np.ndarray,
) -> Callable[
    [float | np.ndarray, float | np.ndarray],
    tuple[float | np.ndarray, float | np.ndarray],
]:
    def _configured_perspective_transform(x, y, z=None):
        denominator = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        x_proj = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denominator
        y_proj = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denominator
        if z is not None:
            return x_proj, y_proj, z
        return x_proj, y_proj

    return _configured_perspective_transform


def _get_pil_perspective_transform(
    H_inv: np.ndarray,
    orig_bounds: tuple[float, float, float, float],
    trans_bounds: tuple[float, float, float, float],
) -> tuple[tuple[int, int], PerspectiveTransform]:
    """Calculates target dimensions and local PIL coefficients for a cropped patch."""
    orig_minx, orig_miny, _, _ = orig_bounds
    trans_minx, trans_miny, trans_maxx, trans_maxy = trans_bounds

    new_width = max(1, int(np.ceil(trans_maxx - trans_minx)))
    new_height = max(1, int(np.ceil(trans_maxy - trans_miny)))

    # T_dst maps local destination (0, 0) -> global destination
    T_dst = np.array(
        [
            [1.0, 0.0, trans_minx],
            [0.0, 1.0, trans_miny],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # T_src_inv maps global source -> local source (0, 0)
    T_src_inv = np.array(
        [
            [1.0, 0.0, -orig_minx],
            [0.0, 1.0, -orig_miny],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Local backward mapping: local dst -> global dst -> global src -> local src
    M_local_inv = T_src_inv @ H_inv @ T_dst
    M_norm = M_local_inv / M_local_inv[2, 2]

    pil_coeffs = (
        float(M_norm[0, 0]),
        float(M_norm[0, 1]),
        float(M_norm[0, 2]),
        float(M_norm[1, 0]),
        float(M_norm[1, 1]),
        float(M_norm[1, 2]),
        float(M_norm[2, 0]),
        float(M_norm[2, 1]),
    )

    return (new_width, new_height), PerspectiveTransform(pil_coeffs)
