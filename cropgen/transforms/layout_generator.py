from PIL import Image
from cropgen.processing.helpers.helper_to_classes import get_union_rect
from cropgen.processing import Paragraph
from cropgen.processing import AnnotatedPage
from cropgen.transforms import (
    InterparagraphTransform,
    IntraparagraphTransform,
)

from copy import deepcopy


class LayoutGenerator:
    def __init__(
        self,
    ):
        self._transform_index = 0

        self.intra_transforms_to_all: list[tuple[IntraparagraphTransform, int]] = []
        self.intra_transforms_specific: dict[
            int, list[tuple[IntraparagraphTransform, int]]
        ] = dict()

        self.inter_transforms: list[InterparagraphTransform] = []

    def add_intra_to_all(self, *transforms: IntraparagraphTransform):
        self.intra_transforms_to_all.extend(
            (transform, i)
            for i, transform in enumerate(transforms, start=self._transform_index)
        )
        self._transform_index += len(transforms)

    def add_intra_to_one(
        self, transform: IntraparagraphTransform, paragraph_index: int
    ):

        self.intra_transforms_specific[paragraph_index].append(
            (transform, self._transform_index)
        )
        self._transform_index += 1

    def add_inter(self, layout: InterparagraphTransform):
        self.inter_transforms.append(layout)

    def apply(self, ann: AnnotatedPage) -> AnnotatedPage:
        """
        Generates a new AnnotatedPage instance by applying the transforms to an annotated page.
        """

        new_ann = deepcopy(ann)

        if not (
            self.intra_transforms_specific
            or self.inter_transforms
            or self.intra_transforms_to_all
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
            layout.in_place(*new_ann.paragraphs)

        new_ann.refresh_geometric_info()

        polygons = [box.polygon for box in new_ann.lines.values()]
        polygons.extend(paragraph.union_polygon() for paragraph in new_ann.paragraphs)

        x1, y1, x2, y2 = get_union_rect(polygons)

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + 1, int(y2) + 1

        w, h = max(1, x2 - x1), max(1, y2 - y1)
        background = new_ann.background.resize(
            size=(w, h), resample=Image.Resampling.BICUBIC
        )
        new_ann.background = background

        return new_ann
