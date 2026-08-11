import cv2
import numpy as np
from PIL import Image


def extract_strokes(
    image: Image.Image,
    background_diameter: int = 35,
    threshold: float = 8.0,
    min_area: int = 3,
    max_area: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (background_diameter, background_diameter),
    )

    background = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel,
    )

    residual = cv2.subtract(background, img)

    _, mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small_kernel,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    areas = stats[:, cv2.CC_STAT_AREA]
    valid_labels = (areas >= min_area) & (areas <= max_area)
    valid_labels[0] = False

    conn = np.zeros(num_labels, dtype=np.uint8)
    conn[valid_labels] = 255

    clean_mask = conn[labels]  # ty: ignore[invalid-argument-type]

    stroke_residual = cv2.bitwise_and(residual, residual, mask=clean_mask)

    return (
        background,
        stroke_residual,
        clean_mask,
    )


def separate_background_and_stroke(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image]:

    _, strokes, mask = extract_strokes(image, 15)

    clean_background = cv2.inpaint(np.asarray(image), mask, 1, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(clean_background), Image.fromarray(strokes)
