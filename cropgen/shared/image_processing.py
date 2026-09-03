from shapely.geometry import Polygon
import cv2
import numpy as np
from cropgen.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from shapely.geometry import Polygon


def extract_strokes(
    page_image_array: np.ndarray,
    background_diameter: int = 15,
    small_diameter: int | None = None,  # 1 or 3 before
    threshold: float = 8.0,
    min_area: int = 3,
    # max_area: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if page_image_array.dtype != np.uint8:
        page_image_array = page_image_array.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (background_diameter, background_diameter),
    )

    background = cv2.morphologyEx(
        page_image_array,
        cv2.MORPH_CLOSE,
        kernel,
    )

    residual = cv2.subtract(background, page_image_array)

    _, mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

    if small_diameter is not None:
        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (small_diameter, small_diameter),
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
    valid_labels = areas >= min_area  # & (areas <= max_area)
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


def extract_border_mask(
    image_array: np.ndarray,
    dark_threshold: int = 50,
    border_margin: int = 10,
) -> np.ndarray:
    """Detects dark connected components attached to or near the image edges."""
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8
    )

    h, w = gray.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)

    for label in range(1, num_labels):
        x, y, comp_w, comp_h, _ = stats[label]

        touches_edge = (
            x <= border_margin
            or y <= border_margin
            or (x + comp_w) >= (w - border_margin)
            or (y + comp_h) >= (h - border_margin)
        )

        if touches_edge:
            border_mask[labels == label] = 255

    return border_mask


def _resize_by_longest_side(img_array: np.ndarray, M: int) -> np.ndarray:
    scale_factor = M / max(img_array.shape)
    size = (
        int(np.ceil(img_array.shape[0] * scale_factor)),
        int(np.ceil(img_array.shape[1] * scale_factor)),
    )
    if scale_factor < 1:
        return cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
    elif scale_factor > 1:
        return cv2.resize(img_array, size, interpolation=cv2.INTER_CUBIC)
    else:
        return img_array


def separate_background_and_stroke(
    image: np.ndarray,
    out_longest_side: int = DATASET_LONGEST_SIZE_PX,
    processing_longest_side: int = PROCESSING_LONGEST_SIDE_PX,
    # inpaint_longest_side: int,
    *,
    background_diameter: int = 15,
    small_diameter: int | None = None,  #  1
    threshold: float = 8.0,
    min_area: int = 3,
    inpaint_dilation: int = 3,
    inpaint_radius: int = 1,
    # max_area: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separates a black and white page image or scan into its stroke and background components.
    """

    image_array = _resize_by_longest_side(image, processing_longest_side)

    _, strokes, stroke_mask = extract_strokes(
        image_array,
        background_diameter,
        small_diameter,
        threshold,
        min_area,  # , max_area
    )

    border_mask = extract_border_mask(image_array)

    combined_mask = cv2.bitwise_or(stroke_mask, border_mask)

    if inpaint_dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inpaint_dilation * 2 + 1, inpaint_dilation * 2 + 1),
        )
        inpaint_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    else:
        inpaint_mask = combined_mask

    # inpaint_mask = _resize_by_longest_side(inpaint_mask, inpaint_longest_side)
    # image_array = _resize_by_longest_side(image_array, inpaint_longest_side)

    clean_background = cv2.inpaint(
        np.asarray(image_array, dtype=np.uint8),
        inpaint_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    clean_background = _resize_by_longest_side(clean_background, out_longest_side)
    strokes = _resize_by_longest_side(strokes, out_longest_side)

    return clean_background, strokes


def crop_or_resize(
    image: np.ndarray,
    x0: int,
    xf: int,
    y0: int,
    yf: int,
    *,
    can_crop: bool = True,
) -> np.ndarray:
    """
    Transforms an image based on target spans.
    - If can_crop is True and target < original: crops a window of target size.
    - Otherwise, resizes the axis to target size when target != original.
    """
    h_max, w_max = image.shape[:2]
    target_w = max(1, int(xf - x0))
    target_h = max(1, int(yf - y0))

    if can_crop and target_w < w_max:
        if x0 < 0:
            x_start, x_end = 0, target_w
        elif x0 + target_w > w_max:
            x_start, x_end = w_max - target_w, w_max
        else:
            x_start, x_end = x0, x0 + target_w
        processed = image[:, x_start:x_end]
    elif target_w != w_max:
        interp = cv2.INTER_AREA if target_w < w_max else cv2.INTER_CUBIC
        processed = cv2.resize(image, (target_w, image.shape[0]), interpolation=interp)
    else:
        processed = image

    curr_h, curr_w = processed.shape[:2]
    if can_crop and target_h < h_max:
        if y0 < 0:
            y_start, y_end = 0, target_h
        elif y0 + target_h > h_max:
            y_start, y_end = h_max - target_h, h_max
        else:
            y_start, y_end = y0, y0 + target_h
        processed = processed[y_start:y_end, :]
    elif target_h != curr_h:
        interp = cv2.INTER_AREA if target_h < curr_h else cv2.INTER_CUBIC
        processed = cv2.resize(processed, (curr_w, target_h), interpolation=interp)

    return processed


def crop_image_with_polygon(
    image: np.ndarray, polygon: Polygon, crop_to_bounds: bool = True
) -> np.ndarray:
    h, w = image.shape[:2]

    if not crop_to_bounds:
        min_x, min_y, max_x, max_y = 0, 0, w, h
        x1, y1, x2, y2 = 0, 0, w, h
        cropped = image.copy()
    else:
        min_x, min_y, max_x, max_y = map(int, polygon.bounds)
        x1 = max(0, min_x)
        y1 = max(0, min_y)
        x2 = min(w, max_x)
        y2 = min(h, max_y)

        if x1 >= x2 or y1 >= y2:
            return np.empty((0, 0, 4), dtype=np.uint8)

        # we copy to detach the object from the original (instead of using a view)
        cropped = image[y1:y2, x1:x2].copy()

    crop_h, crop_w = cropped.shape[:2]

    # only the crop gets converted
    if cropped.ndim == 2:
        bgra_crop = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGRA)
    elif cropped.shape[2] == 3:
        bgra_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    elif cropped.shape[2] == 4:
        bgra_crop = cropped
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    offset = np.array([x1, y1], dtype=np.int32)
    shifted_exterior = np.array(polygon.exterior.coords, dtype=np.int32) - offset

    local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(local_mask, [shifted_exterior.reshape((-1, 1, 2))], color=255)

    if polygon.interiors:
        shifted_interiors = [
            (np.array(interior.coords, dtype=np.int32) - offset).reshape((-1, 1, 2))
            for interior in polygon.interiors
        ]
        cv2.fillPoly(local_mask, shifted_interiors, color=0)

    bgra_crop[:, :, 3] = local_mask

    return bgra_crop
