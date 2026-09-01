from typing import Sequence
from shapely import Polygon
import numpy as np


def calculate_reading_angle(polygon: Polygon) -> float:
    """
    Calcula el ángulo "de lectura" de una caja fijando el lado más largo de su rectángulo delimitador mínimo.
    """
    min_rect = polygon.minimum_rotated_rectangle

    coords = list(min_rect.exterior.coords)[:-1]

    dx_a = coords[1][0] - coords[0][0]
    dy_a = coords[1][1] - coords[0][1]
    len_a = np.hypot(dx_a, dy_a)

    dx_b = coords[2][0] - coords[1][0]
    dy_b = coords[2][1] - coords[1][1]
    len_b = np.hypot(dx_b, dy_b)

    if len_a >= len_b:
        dx, dy = dx_a, dy_a
    else:
        dx, dy = dx_b, dy_b

    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg


def get_union_rect(polys: Sequence[Polygon]) -> tuple[float, float, float, float]:
    """
    Dada una lista coordenadas de cajas imagen con el formato
    (x1, y1, x2, y2), devuelve la bounding box que las contiene a todas.
    """
    if not polys:
        raise ValueError(
            "Cannot extract a union rectangle from an empty sequence of polygons."
        )
    x1 = min(p.bounds[0] for p in polys)
    y1 = min(p.bounds[1] for p in polys)
    x2 = max(p.bounds[2] for p in polys)
    y2 = max(p.bounds[3] for p in polys)
    return x1, y1, x2, y2
