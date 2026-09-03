import cv2
from cropgen.shared.parameters import Parameter
from cropgen.processing.annotated_page import AnnotatedPage
from typing import Optional
import numpy as np


def mother_image(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    width: int = width if width is not None else int(np.random.randint(5, 20))
    height: int = height if height is not None else int(np.random.randint(5, 20))

    arr = np.random.randint(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8,
    )
    return arr


def mother_annotated_page(
    *,
    stroke_img: np.ndarray | None = None,
    background_img: np.ndarray | None = None,
    n_lines_per_paragraph: int = 4,
    n_paragraphs: int = 2,
    line_separator: str = "\n",
) -> AnnotatedPage:
    shape = (
        stroke_img.size
        if stroke_img is not None
        else background_img.size if background_img is not None else (700, 100)
    )
    stroke_img = (
        np.random.randint(low=0, high=128, size=shape, dtype=np.uint8)
        if stroke_img is None
        else stroke_img
    )
    background_img: np.ndarray = (
        np.full(shape, 255, dtype=np.uint8)
        if background_img is None
        else background_img
    )

    polygons: list[list[tuple[float, float]]] = []
    v_i = 0
    v_T = (n_lines_per_paragraph + 1) * n_paragraphs - 1
    for par_i in range(n_paragraphs):
        for _ in range(n_lines_per_paragraph):
            polygons.append(
                [
                    (10, 20 / v_T * v_i),
                    (90, 20 / v_T * v_i),
                    (90, 20 / v_T * (v_i + 1)),
                    (10, 20 / v_T * (v_i + 1)),
                ]
            )

            v_i += 1
        v_i += 1
    totlines = n_lines_per_paragraph * n_paragraphs

    return AnnotatedPage(
        transcriptions=[
            f"Sample transcription {i} for paragraph {j}"
            for j in range(n_paragraphs)
            for i in range(n_lines_per_paragraph)
        ],
        polygon_coords=polygons,
        line_ids=[str(i) for i in range(totlines)],
        rotations=[0 for _ in range(totlines)],
        task_id=999,
        page="999",
        stroke=stroke_img,
        background=background_img,
        line_separtor=line_separator,
    )
