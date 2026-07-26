from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.processing.layout_generator.layouts import (
    InterparagraphTransform,
    IntraparagraphTransform,
)

from copy import deepcopy


class LayoutGenerator:
    def __init__(self, ann: AnnotatedPage):
        self.ann = ann
        self.paragraphs = deepcopy(ann.paragraphs)
        self.intra_layouts: list[list[IntraparagraphTransform]] = [] * len(
            self.paragraphs
        )  # those layouts (to compose its own sequence for each paragraph) that will be applied to each paragraph
        self.inter_layouts: list[
            InterparagraphTransform
        ]  # those layouts (to compose) that will be applied to each paragraph

    # se podría dar con un replace...

    def intra_to_all(self, layout: IntraparagraphTransform):

        for i in range(len(self.intra_layouts)):
            self.intra_layouts[i].append(layout)

    def intra_to_one(self, layout: IntraparagraphTransform, paragraph: Paragraph):

        self.intra_layouts[self.paragraphs.index(paragraph)].append(layout)

    def inter(self, layout: InterparagraphTransform):
        self.inter_layouts.append(layout)

    def apply(self) -> AnnotatedPage:

        for p_layouts, paragraph in zip(self.intra_layouts, self.paragraphs):
            for layout in p_layouts:
                layout(paragraph)

        for layout in self.inter_layouts:
            layout(*self.paragraphs)

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
