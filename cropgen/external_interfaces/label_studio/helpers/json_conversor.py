from shapely.geometry import Polygon
from cropgen.shared.LSTypedDicts.values import PolygonValue
from cropgen.shared.LSTypedDicts.results import (
    RectangleResult,
    PolygonResult,
    RelationResult,
)
from PIL import Image
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedResultItem,
    SimplifiedTextCorrectionResult,
)
import numpy as np
import math


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
    Extrae una imagen (sea rectángulo rotado o polígono arbitrario). Devuelve:
    - residual_crop: el recorte correspondiente del residuo.
    - polygon: el polígono que corresponde a la región.
    - rotation: rotación (en grados) de nuestra región. Si era un rectángulo, es la rotación manual, si no se calcula usando heurísticos.
    - polygon_tool: booleano que representa si la región se hizo usando la herramienta polígono (True) o no.
    """
    if isinstance(
        result, PolygonResult
    ):  # es un polígono (hecho con la herramienta polígono específicamente)
        points = result.value.points

        assert all(len(point) == 2 for point in points)

        return [(point[0], point[1]) for point in points]

    result: RectangleResult

    x = result.value.x
    y = result.value.y
    w = result.value.width
    h = result.value.height
    rotation = result.value.rotation

    if rotation == 0:
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return corners

    theta_rad = math.radians(rotation)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    cx = x + (w / 2.0) * cos_t - (h / 2.0) * sin_t
    cy = y + (w / 2.0) * sin_t + (h / 2.0) * cos_t

    wx = (w / 2.0) * cos_t
    wy = (w / 2.0) * sin_t
    hx = -(h / 2.0) * sin_t
    hy = (h / 2.0) * cos_t

    return [
        (cx - wx - hx, cy - wy - hy),  # arriba-izquierda
        (cx + wx - hx, cy + wy - hy),  # arriba-derecha
        (cx + wx + hx, cy + wy + hy),  # abajo-derecha
        (cx - wx + hx, cy - wy + hy),  # abajo-izquierda
    ]
