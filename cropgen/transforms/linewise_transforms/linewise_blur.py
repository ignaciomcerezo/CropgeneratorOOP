from cropgen.shared.parameters import Parameter
from PIL import Image, ImageFilter

from cropgen.processing.line import Line
from cropgen.transforms.transforms import LinewiseTransform
from shapely.geometry import Polygon


class Blur(LinewiseTransform):
    def __init__(self, radius: Parameter | float = 2.0):
        self.radius: Parameter = Parameter(radius)

    def __call__(
        self, image: Image.Image, polygon: Polygon
    ) -> tuple[Image.Image, Polygon]:
        return (
            image.filter(ImageFilter.GaussianBlur(self.radius())),
            polygon,
        )
