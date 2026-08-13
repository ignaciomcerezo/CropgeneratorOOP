from PIL import Image
from cropgen.processing.helpers.helper_to_classes import get_union_rect
from cropgen.transforms.interparagraph_transforms import (
    Refresh,
)
from cropgen.processing import Paragraph
from cropgen.processing import AnnotatedPage
from cropgen.transforms import (
    InterparagraphTransform,
    IntraparagraphTransform,
)

from copy import deepcopy


class LayoutGenerator:
    def __init__(self, ann: AnnotatedPage):
        self.ann = ann
        self.paragraphs = deepcopy(ann.paragraphs)

        # FIX: Use a list comprehension to create N empty lists
        self.intra_transforms: list[list[IntraparagraphTransform]] = [
            [] for _ in range(len(self.paragraphs))
        ]

        self.inter_transforms: list[InterparagraphTransform] = []

    def add_intra_to_all(self, *transforms: IntraparagraphTransform):
        for transform in transforms:
            for i in range(len(self.intra_transforms)):
                self.intra_transforms[i].append(transform)

    def add_intra_to_one(
        self, transform: IntraparagraphTransform, paragraph: Paragraph
    ):

        self.intra_transforms[self.paragraphs.index(paragraph)].append(transform)

    def add_inter(self, layout: InterparagraphTransform):
        self.inter_transforms.append(layout)

    def apply(self) -> AnnotatedPage:
        if not self.intra_transforms and not self.inter_transforms:
            return self.ann

        for p_layouts, paragraph in zip(self.intra_transforms, self.paragraphs):
            for layout in p_layouts:
                layout(paragraph)

        for layout in self.inter_transforms:
            layout(*self.paragraphs)

        Refresh()(*self.paragraphs)

        polygons = [box.polygon for box in self.ann.lines.values()]
        polygons.extend(paragraph.union_polygon() for paragraph in self.paragraphs)

        x1, y1, x2, y2 = get_union_rect(polygons)

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + 1, int(y2) + 1

        w, h = max(1, x2 - x1), max(1, y2 - y1)
        background = self.ann.background.resize(
            size=(w, h), resample=Image.Resampling.BICUBIC
        )

        return AnnotatedPage.from_paragraphs(
            paragraphs=self.paragraphs,
            task_id=self.ann.task_id,
            background=background,
            completer=self.ann.completer,
            last_update_time=self.ann.last_update_time,
            updater=self.ann.updater,
            annotation_unique_id=hash(
                " ".join(p.transcription() for p in self.paragraphs)
            ),
            line_separator=self.ann.line_separator,
            process_images=self.ann.process_images,
        )
