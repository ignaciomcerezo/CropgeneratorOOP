from shapely import unary_union
import cv2
from cropgen.shared.geometry_processing import get_union_rect
from cropgen.transforms.transforms import (
    LinewiseTransform,
    IntraparagraphFromLinewiseTransform,
)
from typing import Literal
from cropgen.transforms.interparagraph_transforms.avoid_paragraph_intersections import (
    AvoidParagraphIntersections,
)
from cropgen.transforms.intraparagraph_transforms.avoid_line_intersections import (
    AvoidLineIntersections,
)
from collections import defaultdict
from cropgen.ocr_units import OCRParagraph, OCRPage
from cropgen.transforms import (
    InterparagraphTransform,
    IntraparagraphTransform,
)
import numpy as np
from shapely.affinity import translate
from copy import deepcopy


class LayoutGenerator:
    """
    Applies a series of transform to an annotated page to generate a new one.
    Used in LayoutOCRDataset.
    """

    def __init__(
        self,
        avoid_intersections: bool = True,
    ):
        self._transform_index = 0

        self.intra_transforms_to_all: list[tuple[IntraparagraphTransform, int]] = []
        self.intra_transforms_specific: dict[
            int, list[tuple[IntraparagraphTransform, int]]
        ] = defaultdict(lambda: list())

        self.inter_transforms: list[InterparagraphTransform] = []
        self._avoid_intersections = avoid_intersections

        self.ali = AvoidLineIntersections(0.5)
        self.api = AvoidParagraphIntersections(0.5)

    def add_transform(
        self,
        *transforms: InterparagraphTransform
        | IntraparagraphTransform
        | LinewiseTransform,
        scope: Literal["all"] | int = "all",
    ):
        for transform in transforms:
            self._validate_transforms(transform, scope)

        for transform in transforms:
            if isinstance(transform, InterparagraphTransform):
                self._add_inter(transform)
                return

            transform = (
                transform
                if isinstance(transform, IntraparagraphTransform)
                else IntraparagraphFromLinewiseTransform(transform)
            )
            if scope == "all":
                self._add_intra_to_all(transform)
            else:
                self._add_intra_to_one(transform, scope)

    @staticmethod
    def _validate_transforms(transform, scope: Literal["all"] | int):
        if isinstance(transform, InterparagraphTransform) and scope != "all":
            raise ValueError(
                "Cannot pass instances of InterparagraphTransforms when scope != 'all'."
            )
        if not isinstance(
            transform,
            (IntraparagraphTransform, InterparagraphTransform, LinewiseTransform),
        ):
            raise ValueError(
                "Can only use instances of InterparagraphTransform, IntraparagraphTransform or LinewiseTransform."
            )

    def _add_intra_to_all(self, *transforms: IntraparagraphTransform):
        self.intra_transforms_to_all.extend(
            (transform, i)
            for i, transform in enumerate(transforms, start=self._transform_index)
        )
        self._transform_index += len(transforms)

    def _add_intra_to_one(
        self, transform: IntraparagraphTransform, paragraph_index: int
    ):

        self.intra_transforms_specific[paragraph_index].append(
            (transform, self._transform_index)
        )
        self._transform_index += 1

    def _add_inter(self, layout: InterparagraphTransform):
        self.inter_transforms.append(layout)

    def apply(self, ann: OCRPage) -> OCRPage:
        """
        Generates a new AnnotatedPage instance by applying the transforms to an annotated page.
        """

        new_ann = deepcopy(ann)

        if not (
            self.intra_transforms_specific
            or self.inter_transforms
            or self.intra_transforms_to_all
            or self._avoid_intersections
        ):
            return new_ann

        for p_idx, paragraph in enumerate(new_ann.paragraphs):
            transforms_p = self.intra_transforms_to_all.copy()

            if p_idx in self.intra_transforms_specific:
                transforms_p += self.intra_transforms_specific[p_idx]

            transforms_p = sorted(transforms_p, key=lambda x: x[1])

            for transform, _ in transforms_p:
                transform.in_place(paragraph)

        for layout in self.inter_transforms:
            layout.in_place(new_ann.paragraphs)

        self.refresh_annotations_geometric_info(new_ann)

        polygons = [box.polygon for box in new_ann.lines.values()]
        polygons.extend(
            unary_union([line.polygon for line in paragraph.lines])
            for paragraph in new_ann.paragraphs
        )

        if self._avoid_intersections:
            for paragraph in new_ann.paragraphs:
                self.ali.in_place(paragraph)

            self.api.in_place(new_ann.paragraphs)

        x1, y1, x2, y2 = get_union_rect(polygons)

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + 1, int(y2) + 1

        w, h = max(1, x2 - x1), max(1, y2 - y1)
        background = cv2.resize(
            new_ann.background, (w, h), interpolation=cv2.INTER_CUBIC
        )
        new_ann.background = background

        return new_ann

    @staticmethod
    def refresh_annotations_geometric_info(annotation: OCRPage) -> None:
        """
        Refreshes the geometric information of a page to not cause errors. Useful after applying transforms.
        """
        if not annotation.paragraphs:
            return

        if not annotation.lines:
            return

        min_x = min(line.polygon.bounds[0] for line in annotation.lines.values())
        min_y = min(line.polygon.bounds[1] for line in annotation.lines.values())

        for paragraph in annotation.paragraphs:
            for line in paragraph:
                line.polygon = translate(line.polygon, xoff=-min_x, yoff=-min_y)

        for paragraph in annotation.paragraphs:
            total_area = sum(line.polygon.area for line in annotation.lines.values())
            paragraph.avg_rotation = (
                1
                / total_area
                * sum(line.rotation * line.polygon.area for line in paragraph)
            )
            shape = paragraph[0].polygon

            for line in [paragraph[i] for i in range(1, len(paragraph))]:
                shape = shape.union(line.polygon)

            paragraph.centroid = (  # ty: ignore[invalid-assignment]
                shape.centroid.x,
                shape.centroid.y,
            )

            theta_rad = -np.radians(-paragraph.avg_rotation)
            cos_theta = float(np.cos(theta_rad))
            sin_theta = float(np.sin(theta_rad))

            cx_para = float(paragraph.centroid[0])
            cy_para = float(paragraph.centroid[1])

            for line in paragraph:
                cx, cy = line.centroid()
                dx = cx - cx_para
                dy = cy - cy_para

                corrected_x = dx * cos_theta - dy * sin_theta + cx_para
                corrected_y = dx * sin_theta + dy * cos_theta + cy_para

                line.corrected_centroid = (corrected_x, corrected_y)
