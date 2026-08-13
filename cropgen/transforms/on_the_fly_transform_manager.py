from cropgen.transforms.transforms import LinewiseTransform
from cropgen.transforms import IntraparagraphTransform


class OnTheFlyTransformManager:
    def __init__(self):
        self.linewise: list[LinewiseTransform] = []
        self.intraparagraph: list[IntraparagraphTransform] = []

    def add_linewise(self, transform: LinewiseTransform):
        self.linewise.append(transform)

    def add_intraparagraph(self, transform: IntraparagraphTransform):
        self.intraparagraph.append(transform)

    # TODO: pending a .apply() method of some sort.
