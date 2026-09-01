from shapely.geometry import Polygon
from cropgen.external_interfaces.label_studio.ls_typed_dicts import (
    PolygonValue,
    RectangleResult,
    PolygonResult,
    RelationResult,
    SimplifiedResultItem,
    SimplifiedTextCorrectionResult,
)
from PIL import Image
import numpy as np
import math


def pair_lines(
    results: list[SimplifiedResultItem],
) -> tuple[
    dict[str, str],
    dict[str, RectangleResult | PolygonResult],
    dict[str, SimplifiedTextCorrectionResult],
]:
    """
    Generates all Line instances from the Label Studio results.
    """
    id2boxres: dict[str, RectangleResult | PolygonResult] = {
        r.id: r for r in results if isinstance(r, (RectangleResult, PolygonResult))
    }

    id2txtres: dict[str, SimplifiedTextCorrectionResult] = {
        r.id: r for r in results if isinstance(r, SimplifiedTextCorrectionResult)
    }

    box2text: dict[str, str] = dict()

    def is_fragment_with_error(identifyer: str) -> bool:
        if identifyer in id2txtres:
            return True
        elif identifyer in id2boxres:
            return False
        else:
            raise ValueError(f"A relation connects a non-box non-fragment object.")

    seen_boxes: set[str] = set()
    seen_fragments: set[str] = set()
    for r in results:
        if isinstance(r, RelationResult):  # if the result is a relation
            source_id, target_id = r.from_id, r.to_id

            source_is_fragment = is_fragment_with_error(source_id)
            target_is_fragment = is_fragment_with_error(target_id)

            match (source_is_fragment, target_is_fragment):
                case (False, True):
                    box_id, txt_id = source_id, target_id
                case (True, False):
                    txt_id, box_id = source_id, target_id
                case _:
                    # error: box to box OR fragment to fragment association
                    obj_type = ["box", "fragment"][source_is_fragment]
                    raise ValueError(
                        f"(Task {obj_type} to {obj_type} association:"
                        f"{obj_type} {source_id} -> {obj_type} {target_id}."
                    )

            if box_id in seen_boxes:
                raise ValueError(
                    f"(Task box {box_id} has multiple associated fragments."
                )
            if txt_id in seen_fragments:
                raise ValueError(
                    f"(Task fragment {txt_id} has multiple associated boxes."
                )

            seen_boxes.add(box_id)
            seen_fragments.add(txt_id)

            boxres = id2boxres[box_id]
            txtres = id2txtres[txt_id]

            box2text[box_id] = txt_id

    return box2text, id2boxres, id2txtres


def extract_bounds(
    result: PolygonResult | RectangleResult,
) -> list[tuple[float, float]]:
    """
    From a LabelStudio polygon or rectangle result, returns a list of vertices
    of the polygon, in the same percent-of-image units LS uses for polygon points.
    """
    if isinstance(result, PolygonResult):
        points = result.value.points
        assert all(len(point) == 2 for point in points)
        return [(point[0], point[1]) for point in points]

    x = result.value.x
    y = result.value.y
    w = result.value.width
    h = result.value.height
    rotation = result.value.rotation

    ow = result.original_width
    oh = result.original_height

    if rotation == 0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    x_px, y_px = x / 100.0 * ow, y / 100.0 * oh
    w_px, h_px = w / 100.0 * ow, h / 100.0 * oh

    theta = np.radians(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    local_corners = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
    rotated_px = local_corners @ R.T + np.array([x_px, y_px])

    rotated_pct = rotated_px / np.array([ow, oh]) * 100.0

    return [tuple(pt) for pt in rotated_pct]
