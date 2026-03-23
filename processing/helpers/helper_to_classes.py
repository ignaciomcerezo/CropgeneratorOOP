from PIL import Image, ImageDraw
import math
import hashlib  # para los identificadores únicos de subgrafos
from shapely import Polygon, box as boxshape
import numpy as np


def reemplazar_latex_espaciado(texto, apertura, cierre="}"):
    """
    Función para eliminar algunos entornos, por ser generalmente aquellos que Malgoire
    añadía sin necesidad, como {\bf ...} cuando quería enfatizar algo. Algunos
    se correspondían con subrayado, pero como Malgoire no es consistente con
    cuál emplea para indicarlo, es poco realista exigirle al modelo ser capaz de
    predecirlos.
    """
    start_search_pos = 0

    # calculamos el balance inicial
    balance_inicial = apertura.count("{") - apertura.count("}")

    while True:
        # buscamos un momento con balance 0.

        idx_inicio = texto.find(apertura, start_search_pos)

        if (
            idx_inicio == -1
        ):  # si no se encuentra nada, hemos terminado con los reemplazos.
            break

        idx_contenido = idx_inicio + len(apertura)
        balance = balance_inicial
        idx_fin = -1
        i = idx_contenido

        # balanceo de llaves
        while i < len(texto):
            char = texto[i]
            if char == "\\":
                i += 2
                continue
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
                if balance == 0:
                    idx_fin = i
                    break
            i += 1

        if idx_fin != -1:
            # obtenemos la parte anterior sin tocar sus espacios
            parte_anterior = texto[:idx_inicio]

            # obtenemos el contenido interno.
            raw_content = texto[idx_contenido:idx_fin]

            # limpieza básica si usamos sintaxis tipo "{\bf" o "{ \bf", puesto
            # que suele haber espacios ahí, que no queremos
            if apertura.strip().endswith(
                ("bf", "it", "sl")
            ):  # entornos del tipo {\bf ...}.
                contenido_interno = (
                    raw_content.lstrip()
                )  # quitamos el espacio izq solo para estos
            else:
                # para entornos que no son del tipo {\xy ...} sino \xy{...}, mantenemos
                # el contenido interno exacto
                contenido_interno = raw_content

            # reconstruimos el texto.
            texto = parte_anterior + contenido_interno + texto[idx_fin + 1 :]

            # volvemos a empezar la búsqueda
            start_search_pos = len(parte_anterior)
        else:
            # si no hemos encontrado cierre, avanzamos la búsqueda a la siguiente
            # posición
            start_search_pos = idx_inicio + 1

    return texto


def get_deterministic_id(text):
    """
    Genera un identificador 'único' (módulo colisión de hash) y determinista
    a partir de un texto dado usando SHA-256.
    """
    hash_object = hashlib.sha256(text.encode("utf-8"))
    return hash_object.hexdigest()[:8]


def unrotate_image(img, rotation_degrees):
    """
    Des-rota una imagen, quitando también la máscara transparente.
    """
    unrotated = img.rotate(rotation_degrees, resample=Image.BICUBIC, expand=True)

    bbox = unrotated.getbbox()  # solamente la parte no transparente

    if bbox:
        return unrotated.crop(bbox)

    return unrotated


def get_dominant_color(pil_img):
    """
    Calcula el color dominante de una imagen. Realiza el siguiente proceso:
    - Reduce la imagen para que quepa en 50 x 50 píxeles manteniendo las proporciones.
    - Reduce la cantidad de colores a los 5 dominantes, cuantizándola.
    - Devuelve el más común.
    Se emplea para usar como color de fondo en recortes de forma rápida.
    """
    try:
        img_copy = pil_img.copy()
        img_copy.thumbnail((50, 50), resample=Image.Resampling.BICUBIC)

        paletted = img_copy.quantize(colors=5)
        colors = paletted.getcolors()

        if not colors:
            print("Error postcuantización de la imagen")
            return (255, 255, 255)

        dominant_count, dominant_index = max(colors, key=lambda x: x[0])

        palette = paletted.getpalette()
        start = dominant_index * 3
        return tuple(palette[start : start + 3])

    except Exception as E:
        print(f"Error durante la cuantización de la imagen - {E}")
        return (255, 255, 255)


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


def get_rotated_region(val, W, H, img, bg_color):
    """
    Extrae una imagen (sea rectángulo rotado o polígono arbitrario). Devuelve:
    - crop: el recorte de imagen correspondiente.
    - polygon: el polígono que corresponde a la región.
    - rotation: rotación (en grados) de nuestra región. Si era un rectángulo, es la rotación manual, si no se calcula usando heurísticos.
    - polygon_tool: booleano que representa si la región se hizo usando la herramienta polígono (True) o no.
    """

    points = val.get("points")

    if points:  # es un polígono (hecho con la herramienta polígono específicamente)
        # convertimos puntos relativos (0-100) a absolutos (píxeles)
        # Label Studio devuelve [[x1, y1], [x2, y2], ...] si se hizo con un polígono
        abs_points = [(p[0] * W / 100.0, p[1] * H / 100.0) for p in points]

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
            return None

        # recorte rectangular básico
        raw_crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # aplicar Máscara exacta del polígono
        # Creamos una imagen en blanco/negro del tamaño del recorte para usar de máscara alpha
        mask = Image.new("L", raw_crop.size, 0)  # 0 = transparente
        draw = ImageDraw.Draw(mask)

        # Ajustamos los puntos del polígono para que sean relativos al recorte (0,0 es la esquina del recorte)
        local_points = [(p[0] - crop_x1, p[1] - crop_y1) for p in abs_points]

        # dibujamos el polígono relleno en blanco (255 = opaco)
        draw.polygon(local_points, fill=255)
        calculated_rotation = calculate_reading_angle(poly)

        final_image = raw_crop.convert("RGBA")
        final_image.putalpha(mask)

        return final_image, poly, calculated_rotation, True

    # hecho con la herramienta caja-imagen rectangular

    x_pct = val.get("x")
    y_pct = val.get("y")
    w_pct = val.get("width")
    h_pct = val.get("height")
    rotation = val.get("rotation", 0)

    if None in (x_pct, y_pct, w_pct, h_pct):
        return None

    # conversión a píxeles
    x = x_pct * W / 100.0
    y = y_pct * H / 100.0
    w = w_pct * W / 100.0
    h = h_pct * H / 100.0

    # calculamos la forma geométrica usando la función auxiliar
    poly, corners = calculate_polygon(x, y, w, h, rotation)

    # si no hay rotación, el recorte es directo
    if rotation == 0:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        return img.crop((x1, y1, x2, y2)), poly, 0, False

    # si hay rotación, usamos los vértices calculados para definir la bounding box del recorte
    all_x = [p[0] for p in corners]
    all_y = [p[1] for p in corners]

    pad = 0
    crop_x1 = int(math.floor(min(all_x))) - pad
    crop_y1 = int(math.floor(min(all_y))) - pad
    crop_x2 = int(math.ceil(max(all_x))) + pad
    crop_y2 = int(math.ceil(max(all_y))) + pad

    # Validaciones de límites
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(W, crop_x2)
    crop_y2 = min(H, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    # Recorte inicial
    raw_crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Creación de máscara
    mask = Image.new("L", raw_crop.size, 0)
    draw = ImageDraw.Draw(mask)

    # Convertir coordenadas globales a locales para la máscara
    local_corners = [(p[0] - crop_x1, p[1] - crop_y1) for p in corners]

    draw.polygon(local_corners, fill=255)

    final_image = raw_crop.convert("RGBA")
    final_image.putalpha(mask)

    return final_image, poly, rotation, False


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

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg


def get_union_rect(polys):
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
    return (x1, y1, x2, y2)


def get_connected_components(adj):
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


def compose_collage(image_boxes, fill_color):
    # calculamos la región mínima de la imagen que contiene todas las cajas
    X1, Y1, X2, Y2 = get_union_rect([box.polygon for box in image_boxes])

    # Convertimos a enteros (Floor para arriba-izq, Ceil para abajo-der para asegurar cobertura)
    X1, Y1 = int(X1), int(Y1)
    X2, Y2 = int(X2) + 1, int(Y2) + 1

    crop_width, crop_height = X2 - X1, Y2 - Y1

    # creamos el collage
    mode = (
        "RGBA" if len(fill_color) == 4 else "RGB"
    )  # si tiene transparencia, usamos RGBA
    collage = Image.new(mode, (crop_width, crop_height), tuple(fill_color))

    for box in image_boxes:
        box_x0, box_y0, _, _ = box.polygon.bounds

        # calculamos la posición relativa al nuevo lienzo
        paste_x, paste_y = int(box_x0 - X1), int(box_y0 - Y1)

        if box.crop.mode == "RGBA":
            # usamos la propia imagen como máscara de transparencia
            collage.paste(box.crop, (paste_x, paste_y), mask=box.crop)
        else:
            collage.paste(box.crop, (paste_x, paste_y))

    return collage


def subdictionary(nodes, adj) -> dict[str, set[str]]:
    subdict = {}
    for node in nodes:
        subdict[node] = adj[node]
    return subdict
