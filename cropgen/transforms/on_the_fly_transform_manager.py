from shapely.geometry import Polygon
from cropgen.transforms.transforms import LinewiseTransform
from cropgen.transforms import IntraparagraphTransform
from numpy.random import rand
from PIL import Image


class OCROnTheFlyTransformPack:
    def __init__(self):
        self._linewise: list[LinewiseTransform] = []
        self._linewise_p: list[float] = []
        self._intraparagraph: list[IntraparagraphTransform] = []
        self._intraparagraph_p: list[float] = []

    def add_linewise(self, transform: LinewiseTransform, probability: float = 1):
        if (probability > 1) or (probability < 0):
            raise ValueError("probability must be between 0 and 1")
        self._linewise.append(transform)
        self._linewise_p.append(probability)

    def add_intraparagraph(
        self, transform: IntraparagraphTransform, probability: float = 1
    ):
        if (probability > 1) or (probability < 0):
            raise ValueError("probability must be between 0 and 1")
        self._intraparagraph.append(transform)
        self._intraparagraph_p.append(probability)

    # TODO: add collision toggle (solve_collisions) and implement it in .transform()

    def transform(
        self, images: list[Image.Image], polygons: list[Polygon]
    ) -> tuple[list[Image.Image], list[Polygon]]:

        for i, (image, polygon) in enumerate(zip(images, polygons)):
            for linewise, p in zip(self._linewise, self._linewise_p):
                if (p == 1) or ((p <= 1) and (rand() < p)):
                    images[i], polygons[i] = linewise(image, polygon)

        for intraparagraph, p in zip(self._intraparagraph, self._intraparagraph_p):
            if (p == 1) or ((p <= 1) and (rand() < p)):
                images, polygons = intraparagraph((images, polygons))

        return images, polygons
