from PIL import Image, ImageDraw
import math
import hashlib  # para los identificadores únicos de subgrafos
from shapely import Polygon, box as boxshape
import numpy as np
from typing import TYPE_CHECKING, Sequence


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


def get_union_rect(polys: Sequence[Polygon]):
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
