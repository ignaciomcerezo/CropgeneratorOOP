from PIL import Image, ImageDraw
import math
import hashlib  # para los identificadores únicos de subgrafos
from shapely import Polygon, box as boxshape
import numpy as np
from cropgen.shared.LSTypedDicts.values import RectangleValue, PolygonValue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cropgen.processing.ImageBox import ImageBox


def get_deterministic_id(text, length: int = 8):
    """
    Genera un identificador 'único' (módulo colisión de hash) y determinista
    a partir de un texto dado usando SHA-256.
    """
    hash_object = hashlib.sha256(text.encode("utf-8"))
    return hash_object.hexdigest()[:length]


def calculate_polygon(x, y, w, h, rotation):
    """
    Calcula los vértices del polígono rotado y devuelve el objeto polygon de shapely
    y la lista de vértices (para la función get_rotated_region)
    """
    # Caso 1: Sin rotación (Caja alineada al eje)
    if rotation == 0:
        # Definimos esquinas en orden para consistencia (TL, TR, BR, BL)
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        # Usamos box de shapely que es más eficiente para rectángulos simples
        rect = boxshape(x, y, x + w, y + h)
        return rect, corners

    # Caso 2: Con rotación
    theta_rad = math.radians(rotation)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Lógica original de cálculo de centro y vectores
    # Nota: Mantenemos tu lógica exacta para preservar el comportamiento actual
    cx = x + (w / 2.0) * cos_t - (h / 2.0) * sin_t
    cy = y + (w / 2.0) * sin_t + (h / 2.0) * cos_t

    wx = (w / 2.0) * cos_t
    wy = (w / 2.0) * sin_t
    hx = -(h / 2.0) * sin_t
    hy = (h / 2.0) * cos_t

    corners = [
        (cx - wx - hx, cy - wy - hy),  # arriba-izquierda
        (cx + wx - hx, cy + wy - hy),  # arriba-derecha
        (cx + wx + hx, cy + wy + hy),  # abajo-derecha
        (cx - wx + hx, cy - wy + hy),  # abajo-izquierda
    ]

    return Polygon(corners), corners


def calculate_polygon_angle(poly):
    """
    Calcula el ángulo de rotación del polígono basándose en su
    rectángulo mínimo orientado (minimum bounding box de shapely).
    Asume que el lado más largo del rectángulo corresponde a la orientación del texto.
    """
    rect = poly.minimum_rotated_rectangle

    coords = list(rect.exterior.coords)

    p0, p1 = coords[0], coords[1]
    dist_a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    p2 = coords[2]
    dist_b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # determinamos cuál es el lado "largo" (la base del texto)
    if dist_a > dist_b:
        # vector p0 -> p1
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
    else:
        # vector p1 -> p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

    # ángulo en grados
    angle = math.degrees(math.atan2(dy, dx))

    # normalizams teniendo en cuenta que la lectura es de izq. a derecha
    if angle < -45:
        angle += 180
    elif angle > 135:
        angle -= 180

    return angle


def get_rotated_region(
    val: PolygonValue | RectangleValue,
    page_width: float | int,
    page_height: float | int,
    residual: Image.Image,
) -> tuple[Image.Image, Polygon, float, bool]:
    """
    Extrae una imagen (sea rectángulo rotado o polígono arbitrario). Devuelve:
    - residual_crop: el recorte correspondiente del residuo.
    - polygon: el polígono que corresponde a la región.
    - rotation: rotación (en grados) de nuestra región. Si era un rectángulo, es la rotación manual, si no se calcula usando heurísticos.
    - polygon_tool: booleano que representa si la región se hizo usando la herramienta polígono (True) o no.
    """

    def _crop_with_alpha(
        source: Image.Image,
        box: tuple[int, int, int, int],
        mask: Image.Image | None = None,
    ):
        cropped = source.crop(box)
        if mask is None:
            return cropped
        final_image = cropped.convert("RGBA")
        final_image.putalpha(mask)
        return final_image

    if isinstance(
        val, PolygonValue
    ):  # es un polígono (hecho con la herramienta polígono específicamente)
        points = val.points
        # convertimos puntos relativos (0-100) a absolutos (píxeles)
        # Label Studio devuelve [[x1, y1], [x2, y2], ...] si se hizo con un polígono
        abs_points = [
            (p[0] * page_width / 100.0, p[1] * page_height / 100.0) for p in points
        ]

        # 2. Crear objeto Polygon de Shapely (para calcular intersecciones en el grafo después)
        poly = Polygon(abs_points)
        if not poly.is_valid:
            poly = poly.buffer(
                0
            )  # Intento simple de arreglar auto-intersecciones si las hubiera

        # calcular la Bounding Box que encierra el polígono para hacer el recorte inicial
        min_x, min_y, max_x, max_y = poly.bounds

        # Padding opcional
        # pad = 0

        # Convertimos a enteros para el crop (floor para mín, ceil para máx)
        crop_x1 = int(math.floor(min_x))  # - pad
        crop_y1 = int(math.floor(min_y))  # - pad
        crop_x2 = int(math.ceil(max_x))  # + pad
        crop_y2 = int(math.ceil(max_y))  # + pad

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError("Crop width <= 0 or crop height <= 0.")

        # recorte rectangular básico
        box = (crop_x1, crop_y1, crop_x2, crop_y2)

        # aplicar Máscara exacta del polígono
        # Creamos una imagen en blanco/negro del tamaño del recorte para usar de máscara alpha
        mask = Image.new(
            "L", (crop_x2 - crop_x1, crop_y2 - crop_y1), 0
        )  # 0 = transparente
        draw = ImageDraw.Draw(mask)

        # Ajustamos los puntos del polígono para que sean relativos al recorte (0,0 es la esquina del recorte)
        local_points = [(p[0] - crop_x1, p[1] - crop_y1) for p in abs_points]

        # dibujamos el polígono relleno en blanco (255 = opaco)
        draw.polygon(local_points, fill=255)
        calculated_rotation = calculate_reading_angle(poly)

        residual_crop = _crop_with_alpha(residual, box, mask)

        return residual_crop, poly, calculated_rotation, True

    # hecho con la herramienta caja-imagen rectangular

    val: RectangleValue

    x_pct = val.x
    y_pct = val.y
    w_pct = val.width
    h_pct = val.height
    rotation = val.rotation

    # conversión a píxeles
    x = x_pct * page_width / 100.0
    y = y_pct * page_height / 100.0
    w = w_pct * page_width / 100.0
    h = h_pct * page_height / 100.0

    # calculamos la forma geométrica usando la función auxiliar
    poly, corners = calculate_polygon(x, y, w, h, rotation)

    # si no hay rotación, el recorte es directo
    if rotation == 0:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(page_width, x2), min(page_height, y2)

        box = (x1, y1, x2, y2)

        residual_crop = residual.crop(box)
        return residual_crop, poly, 0, False

    # si hay rotación, usamos los vértices calculados para definir la bounding box del recorte
    all_x = [p[0] for p in corners]
    all_y = [p[1] for p in corners]

    pad = 0
    crop_x1 = int(math.floor(min(all_x))) - pad
    crop_y1 = int(math.floor(min(all_y))) - pad
    crop_x2 = int(math.ceil(max(all_x))) + pad
    crop_y2 = int(math.ceil(max(all_y))) + pad

    # comprobamos que el área del polígono no sea 0 ni negativa
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = int(min(page_width, crop_x2))
    crop_y2 = int(min(page_height, crop_y2))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Null height or width.")

    # recorte inicial
    box = (crop_x1, crop_y1, crop_x2, crop_y2)
    mask = Image.new("L", (crop_x2 - crop_x1, crop_y2 - crop_y1), 0)
    draw = ImageDraw.Draw(mask)

    # convertimos a coordenadas locales para la máscara
    local_corners = [(p[0] - crop_x1, p[1] - crop_y1) for p in corners]

    draw.polygon(local_corners, fill=255)

    residual_crop = _crop_with_alpha(residual, box, mask)

    return residual_crop, poly, rotation, False


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


def get_union_rect(polys: list[Polygon]):
    """
    Dada una lista coordenadas de cajas imagen con el formato
    (x1, y1, x2, y2), devuelve la bounding box que las contiene a todas.
    """
    if not polys:
        return None
    x1 = min(p.bounds[0] for p in polys)
    y1 = min(p.bounds[1] for p in polys)
    x2 = max(p.bounds[2] for p in polys)
    y2 = max(p.bounds[3] for p in polys)
    return x1, y1, x2, y2


def get_connected_components(adj: dict[str, set]):
    """
    Dado un grafo de adyacencia, devuelve las componentes conexas como una lista
    de conjuntos de nodos.
    """
    # backtracking habitual no recursivo para generar las componentes conexas de
    # un grafo usando un diccionario
    visited = set()
    components = []

    for v in adj:
        if v not in visited:  # si es la primera vez que vemos este nodo,
            comp = set()
            q = [v]
            while q:
                curr = q.pop(0)
                if curr in visited:
                    continue
                # añadimos el nodo a visitados y a la componente actual
                visited.add(curr)
                comp.add(curr)
                # añadimos los nodos adyacentes al actual a la lista para procesar
                # pues deben estar en la misma componente conexa.
                q.extend(list(adj.get(curr, [])))
            # añadimos la componente conexa
            components.append(comp)
    return components


def compose_collage(
    image_boxes: list["ImageBox"],
    background: Image.Image,
    tight_layout: bool = True,
) -> Image.Image:
    """
    Generates the corresponding collage of lines from the image boxes and a backgroud fill color.
    If min_bounding_boxes is provided, each element is taken to be the coordinates where the leftmost
    topmost point of the bounding box of each line will be placed. If not provided, it takes that
    information from the image_box instances themselves.
    """

    if tight_layout:
        # calculamos la región mínima de la imagen que contiene todas las cajas
        x1, y1, x2, y2 = get_union_rect([box.polygon for box in image_boxes])

        # Convertimos a enteros (Floor para arriba-izq, Ceil para abajo-der para asegurar cobertura)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + 1, int(y2) + 1

        crop_width, crop_height = x2 - x1, y2 - y1
        collage = background.crop((x1, y1, x2, y2))
    else:
        collage = background
        x1 = 0
        y1 = 0

    overlay: np.ndarray = np.full(np.asarray(collage).shape, 0)

    for box in image_boxes:
        box_x0, box_y0, _, _ = box.polygon.bounds

        # calculamos la posición relativa al nuevo lienzo
        paste_x, paste_y = int(box_x0 - x1), int(box_y0 - y1)

        stroke_rgba = np.asarray(box.stroke_crop.convert("RGBA"))

        stroke = stroke_rgba[..., 0]
        alpha = stroke_rgba[..., 3]

        masked_stroke = stroke * (alpha / 255.0)

        overlay[
            paste_y : paste_y + box.stroke_crop.height,
            paste_x : paste_x + box.stroke_crop.width,
        ] += masked_stroke.astype(np.uint8)

    # difference instead of addition as our strokes are reversed in intensity
    collage = Image.fromarray(
        np.clip(np.asarray(collage, dtype=np.float32) - overlay, 0, 255).astype(
            np.uint8
        )
    )

    return collage


def subdictionary(nodes, adj) -> dict[str, set[str]]:
    subdict = {}
    for node in nodes:
        subdict[node] = adj[node]
    return subdict


def is_path_graph(graph_dict):
    """
    checks if a graph is isomorphic to a path graph by checking if it is connected and
    its degree sequence matches that of a path graph.
    """
    n = len(graph_dict)

    if n == 0:
        return False
    if n == 1:
        return len(list(graph_dict.values())[0]) == 0

    degrees = [len(neighbors) for neighbors in graph_dict.values()]

    if degrees.count(1) != 2 or degrees.count(2) != n - 2:
        return False

    visited = set()

    start_node = next(
        node for node, neighbors in graph_dict.items() if len(neighbors) == 1
    )

    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph_dict[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return len(visited) == n
