from cropgen.shared.parameters import Parameter
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import Polygon

from cropgen.processing.line import Line
from cropgen.transforms.transforms import LinewiseTransform


class LinewiseHorizontalStretch(LinewiseTransform):
    def __init__(self, scale_factor: Parameter | float = 1.2):

        self.scale_factor = Parameter(scale_factor)

    def __call__(self, box: Line) -> tuple[Image.Image, Polygon]:
        image = box.stroke_crop
        stretched_image = image.resize(
            (max(1, round(image.width * abs(self.scale_factor()))), image.height),
            resample=Image.Resampling.BILINEAR,
        )

        min_x, min_y, max_x, max_y = box.polygon.bounds
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        stretched_polygon = scale(
            box.polygon,
            xfact=self.scale_factor,
            yfact=1.0,
            origin=center,
        )

        return stretched_image, stretched_polygon
