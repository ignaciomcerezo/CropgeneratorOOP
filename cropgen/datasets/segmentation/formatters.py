from typing import Callable, Any, Sequence
from PIL import Image
from shapely import Polygon, MultiPolygon
import numpy as np
import cv2

_formatter_type = Callable[[Image.Image, Sequence[Polygon]], Any]


def _polygon_to_mask(poly: Polygon | MultiPolygon, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly.is_empty:
        return mask

    if poly.area == 0:
        # we expand the polygon if it has no area
        poly = poly.buffer(1.5)

    if isinstance(poly, Polygon):
        polys = [poly]
    elif isinstance(poly, MultiPolygon):
        polys = poly.geoms
    else:
        polys = []

    for p in polys:
        coords = np.array(p.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [coords], 1)

    return mask
