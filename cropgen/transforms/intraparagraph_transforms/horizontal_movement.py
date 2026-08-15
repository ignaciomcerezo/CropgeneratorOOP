from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from shapely.geometry import Polygon
from cropgen.transforms.transforms import (
    IntraparagraphFromLinewiseTransform,
)
from sympy.stats import Uniform
from cropgen.shared.parameters import (
    Parameter,
    NormalDistribution,
    UniformDistribution,
)
from dataclasses import dataclass
from typing import Literal, Callable, Sequence


from cropgen.processing import Paragraph, Line
from cropgen.transforms import (
    LinewiseTransform,
)
from shapely.affinity import translate
import numpy as np
from PIL import Image

NOISES = Literal["linear", "wave", "from_amplitude_parameter", "zigzag"]

scalar = Parameter | float


def _centroid(poly: Polygon) -> tuple[float, float]:
    return poly.centroid.x, poly.centroid.y


@dataclass
class HorizontalMovement(IntraparagraphFromLinewiseTransform):

    def __init__(
        self,
        noise_type: NOISES,
        period: scalar | None = None,
        amplitude: scalar | None = None,
        slope: scalar | None = None,
        intercept: scalar | None = 0,
    ):
        self.noise_type = noise_type
        self.period = Parameter(period) if period is not None else None
        self.amplitude = Parameter(amplitude) if amplitude is not None else None
        self.slope = Parameter(slope) if slope is not None else None
        self.intercept = Parameter(intercept) if intercept is not None else None
        self.type2map: dict[NOISES, Callable[[list[Polygon]], list[Polygon]]] = {
            "linear": self._call_linear_polygons,
            "wave": self._call_wave_polygons,
            "from_amplitude_parameter": self._call_random_polygons,
            "zigzag": self._call_zigzag_polygons,
        }

    def _call_linear_polygons(self, polygons: list[Polygon]) -> list[Polygon]:

        if self.slope is None:
            raise ValueError(
                "Cannot use linear horizontal movement if no slope is passed."
            )

        if self.intercept is None:
            raise ValueError(
                "Cannot use linear horizontal movement if None is passed as intercept."
            )

        x0, y0 = _centroid(polygons[0])
        for i, polygon in enumerate(polygons):
            xi, yi = _centroid(polygon)
            polygons[i] = translate(
                polygon, xoff=self.intercept() + self.slope() * abs(yi - y0)
            )

        return polygons

    def _call_wave_polygons(self, polygons: list[Polygon]) -> list[Polygon]:

        if self.amplitude is None:
            raise ValueError("Cannot use wave movement if no amplitude is passed.")

        if self.period is None:
            raise ValueError("Cannot use wave movement if no period is passed.")

        _, y0 = _centroid(polygons[0])

        for i, polygon in enumerate(polygons):
            _, yi = _centroid(polygon)

            xoff = self.amplitude() * np.cos(2 * np.pi * (yi - y0) / self.period())

            polygons[i] = translate(polygon, xoff=xoff)

        return polygons

    def _call_zigzag_polygons(self, polygons: list[Polygon]) -> list[Polygon]:
        _, y0 = _centroid(polygons[0])

        if self.amplitude is None:
            raise ValueError("Cannot use zigzag movement if no amplitude is passed.")

        if self.period is None:
            raise ValueError("Cannot use zigzag movement if no period is passed.")

        for i, polygon in enumerate(polygons):
            _, yi = _centroid(polygon)

            phase = int((yi - y0) / self.period())
            xoff = self.amplitude if phase % 2 == 0 else -self.amplitude()

            polygons[i] = translate(polygon, xoff=xoff)

        return polygons

    def _call_random_polygons(self, polygons: list[Polygon]) -> list[Polygon]:
        if self.amplitude is None:
            raise ValueError("Cannot use random movement if no amplitude is passed.")

        for i, polygon in enumerate(polygons):
            xoff = self.amplitude()
            polygons.append(translate(polygon, xoff=xoff))

        return polygons

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph
            | Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)
        polygons = self.type2map[self.noise_type](polygons)

        return images, polygons
