from cropgen.processing.layout_generator.InterparagraphTransforms.refresh import Refresh
from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.processing.layout_generator.transforms import (
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

    def add_intra_to_all(self, transform: IntraparagraphTransform):
        for i in range(len(self.intra_transforms)):
            self.intra_transforms[i].append(transform)

    def add_intra_to_one(
        self, transform: IntraparagraphTransform, paragraph: Paragraph
    ):

        self.intra_transforms[self.paragraphs.index(paragraph)].append(transform)

    def add_inter(self, layout: InterparagraphTransform):
        self.inter_transforms.append(layout)

    def apply(self) -> AnnotatedPage:
        for p_layouts, paragraph in zip(self.intra_transforms, self.paragraphs):
            for layout in p_layouts:
                layout(paragraph)

        for layout in self.inter_transforms:
            layout(*self.paragraphs)

        Refresh()(*self.paragraphs)

        return AnnotatedPage.from_paragraphs(
            self.paragraphs,
            self.ann.task_id,
            self.ann.background_color,
            self.ann.last_update_time,
            self.ann.completer,
            self.ann.updater,
            annotation_unique_id=hash(
                " ".join(p.transcription() for p in self.paragraphs)
            ),
            line_separator=self.ann.line_separator,
            process_images=self.ann.process_images,
        )
