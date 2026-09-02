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
        strength: Parameter | float = 0.2,
        tilt_axis: Literal["vertical", "horizontal"] = "horizontal",
    ):
        self.relative = Parameter(strength)
        assert self.relative.is_bounded(
            -1, 1
        ), "The strength of the tilt must lie be between (-1, 1)."
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

        # The union may be a MultiPolygon, but its minimum rotated rectangle
        # is still a Polygon, so this is fine.
        union = LineGroupInfo.polygon_union(polygons)
        mbr: Polygon = union.minimum_rotated_rectangle

        reading_direction = np.asarray(LGinfo.reading_direction, dtype=float)
        orthogonal_direction = np.asarray(LGinfo.orthogonal_direction, dtype=float)

        # Normalize the directions so that projections are comparable.
        reading_norm = np.linalg.norm(reading_direction)
        orthogonal_norm = np.linalg.norm(orthogonal_direction)

        if reading_norm == 0 or orthogonal_norm == 0:
            raise ValueError("Geometry failed: invalid reading/orthogonal direction.")

        reading_direction /= reading_norm
        orthogonal_direction /= orthogonal_norm

        # The four corners of the MBR are returned in cyclic order.
        points: list[Vector2D] = [
            np.asarray(point, dtype=float) for point in mbr.exterior.coords[:-1]
        ]

        if len(points) != 4:
            raise ValueError(
                f"Geometry failed: expected 4 MBR corners, got {len(points)}."
            )

        # IMPORTANT:
        # Use the MBR's center rather than LGinfo.center. The latter need not be
        # the center of the bounding rectangle, so sign-based quadrant tests can
        # incorrectly put two corners in the same quadrant.
        center = np.asarray(mbr.centroid.coords[0], dtype=float)

        # Project each corner onto the reading and orthogonal axes.
        #
        # reading_projection:
        #   tells us which end of the rectangle the point belongs to.
        #
        # orthogonal_projection:
        #   distinguishes the two corners at that end.
        projected_points = [
            (
                np.dot(reading_direction, point - center),
                np.dot(orthogonal_direction, point - center),
                point,
            )
            for point in points
        ]

        # Split the four corners into the two corners at each end of the
        # reading direction.
        projected_points.sort(key=lambda item: item[0])

        low_reading = projected_points[:2]
        high_reading = projected_points[2:]

        # Within each end, distinguish the two corners using the orthogonal
        # direction.
        #
        # At the "low reading" end:
        #   a = orthogonally negative
        #   b = orthogonally positive
        #
        # At the "high reading" end:
        #   d = orthogonally negative
        #   c = orthogonally positive
        #
        # This gives the cyclic ordering a -> b -> c -> d for the usual
        # orientation of reading_direction / orthogonal_direction.
        low_reading.sort(key=lambda item: item[1])
        high_reading.sort(key=lambda item: item[1])

        a = low_reading[0][2]
        b = low_reading[1][2]
        d = high_reading[0][2]
        c = high_reading[1][2]

        if not self._tilt_horizontal:
            # For a vertical tilt, reinterpret the rectangle's axes so that
            # the same subsequent transformation logic can be used.
            a, b, c, d = a, d, b, c

        t = 0.5 * self.relative()

        a_moved, b_moved = self._symm_lerp(a, b, t)
        c_moved, d_moved = self._symm_lerp(c, d, -t)

        source_points = [a, b, c, d]
        destination_points = [a_moved, b_moved, c_moved, d_moved]

        H_fwd = find_homography_matrix(
            source_points,
            destination_points,
        )

        H_inv = np.linalg.inv(H_fwd)

        shapely_configured_transform = _get_shapely_perspective_transform(H_fwd)

        for i, (image, polygon) in enumerate(zip(images, polygons)):
            orig_bounds = polygon.bounds

            transformed_polygon = transform(
                shapely_configured_transform,
                polygon,
            )

            polygons[i] = transformed_polygon

            trans_bounds = transformed_polygon.bounds

            expected_size, image_configured_transform = _get_pil_perspective_transform(
                H_inv,
                orig_bounds,
                trans_bounds,
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
