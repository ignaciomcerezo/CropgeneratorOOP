from shapely.geometry import Polygon
from typing import Literal, Collection
from PIL import Image
import cv2
import numpy as np


def unrotate_image(img, rotation_degrees) -> Image.Image:
    """
    Des-rota una imagen, quitando también la máscara transparente.
    """
    unrotated = img.rotate(rotation_degrees, resample=Image.BICUBIC, expand=True)

    bbox = unrotated.getbbox()  # solamente la parte no transparente

    if bbox:
        return unrotated.crop(bbox)

    return unrotated


def get_dominant_color(pil_img) -> tuple[int, int, int]:
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
            return 255, 255, 255

        dominant_count, dominant_index = max(colors, key=lambda x: x[0])

        palette: list[int] = paletted.getpalette()
        start = dominant_index * 3
        return (palette[start], palette[start + 1], palette[start + 2])

    except Exception as E:
        print(f"Error durante la cuantización de la imagen - {E}")
        return 255, 255, 255


KERNELS = {
    "diamond": cv2.MORPH_DIAMOND,
    "circle": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
    "rect": cv2.MORPH_RECT,
}

KERNEL_TYPES = Literal["diamond", "circle", "cross", "rect"]

# better version in text_background_separator.py
# def get_background_and_residual(
#     image: Image.Image, kernel_name="rect", diameter: int = 35
# ) -> tuple[Image.Image, Image.Image]:
#     """
#     Extracts the low-frequency background and signed high-frequency residual
#     """
#     img_float = np.array(image, dtype=np.float32)

#     kernel = cv2.getStructuringElement(KERNELS[kernel_name], (diameter, diameter))

#     bg_float = cv2.morphologyEx(img_float, cv2.MORPH_CLOSE, kernel)

#     signed_residual = img_float - bg_float

#     return Image.fromarray(bg_float), Image.fromarray(signed_residual)


def polygon2pts(polygon: Polygon):
    """Converts a Shapely Polygon exterior boundary to OpenCV int32 coordinates."""
    coords = np.array(polygon.exterior.coords, dtype=np.float32)
    return [np.int32(coords)]


def get_polygon_mask(
    image_shape: tuple[int, int], polygons: Collection[Polygon]
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        coords = np.array(poly.exterior.coords[:-1], dtype=np.int32)
        cv2.fillPoly(mask, [coords], 255)
    return mask


def crop_to_polygon(img: Image.Image, poly: Polygon) -> Image.Image:
    """Crops an image according to a polygon"""
    image_arr = np.array(img)
    mask = get_polygon_mask(image_arr.shape, [poly])

    image_arr[mask == 0] = 0

    return Image.fromarray(image_arr)


def inpaint_exterior_line_ring(
    image: Image.Image,
    polygons: Collection[Polygon],
    ring_width: int = 15,
) -> Image.Image:
    image_arr = np.array(image)

    poly_mask = get_polygon_mask(image_arr.shape, polygons)

    kernel_size = 2 * ring_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    dilated_mask = cv2.dilate(poly_mask, kernel)
    exterior_ring_mask = cv2.subtract(dilated_mask, poly_mask)

    inpaint_radius = max(3, ring_width // 2)
    return Image.fromarray(
        cv2.inpaint(
            image_arr,
            exterior_ring_mask,
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )
    )
