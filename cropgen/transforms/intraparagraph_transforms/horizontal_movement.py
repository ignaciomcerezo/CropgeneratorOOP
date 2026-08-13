from shapely.geometry import Polygon
from cropgen.transforms.transforms import (
    IntraparagraphFromLinewiseTransform,
)
from sympy.stats import Uniform
from cropgen.shared.parameters import (
    Parameter,
    NormalDistribution,
    UniformDistribution,
    instanciate_if_parameter,
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
        self.type2map: dict[
            NOISES, Callable[[Paragraph | Sequence[Line]], list[Polygon]]
        ] = {
            "linear": self._call_linear_polygons,
            "wave": self._call_wave_polygons,
            "from_amplitude_parameter": self._call_random_polygons,
            "zigzag": self._call_zigzag_polygons,
        }

    def _call_linear_polygons(
        self, line_group: Paragraph | Sequence[Line]
    ) -> list[Polygon]:

        if self.slope is None:
            raise ValueError(
                "Cannot use linear horizontal movement if no slope is passed."
            )

        if self.intercept is None:
            raise ValueError(
                "Cannot use linear horizontal movement if None is passed as intercept."
            )

        x0, y0 = line_group[0].centroid()
        polygons = []
        for line in line_group:
            xi, yi = line.centroid()
            polygons.append(
                translate(
                    line.polygon, xoff=self.intercept() + self.slope() * abs(yi - y0)
                )
            )
        return polygons

    def _call_wave_polygons(
        self, line_group: Paragraph | Sequence[Line]
    ) -> list[Polygon]:

        if self.amplitude is None:
            raise ValueError("Cannot use wave movement if no amplitude is passed.")

        if self.period is None:
            raise ValueError("Cannot use wave movement if no period is passed.")

        _, y0 = line_group[0].centroid()
        polygons = []
        for line in line_group:
            _, yi = line.centroid()

            xoff = self.amplitude() * np.cos(2 * np.pi * (yi - y0) / self.period())

            polygons.append(translate(line.polygon, xoff=xoff))
        return polygons

    def _call_zigzag_polygons(
        self, line_group: Paragraph | Sequence[Line]
    ) -> list[Polygon]:
        _, y0 = line_group[0].centroid()
        polygons = []

        if self.amplitude is None:
            raise ValueError("Cannot use zigzag movement if no amplitude is passed.")

        if self.period is None:
            raise ValueError("Cannot use zigzag movement if no period is passed.")

        for line in line_group:
            _, yi = line.centroid()

            phase = int((yi - y0) / self.period())
            xoff = self.amplitude if phase % 2 == 0 else -self.amplitude()

            polygons.append(translate(line.polygon, xoff=xoff))

        return polygons

    def _call_random_polygons(
        self, line_group: Paragraph | Sequence[Line]
    ) -> list[Polygon]:
        if self.amplitude is None:
            raise ValueError("Cannot use random movement if no amplitude is passed.")
        polygons = []
        for line in line_group:
            xoff = self.amplitude()
            polygons.append(translate(line.polygon, xoff=xoff))
        return polygons

    def __call__(
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[Polygon]]:
        polygons = self.type2map[self.noise_type](line_group)
        images = [line.stroke_crop for line in line_group]

        return images, polygons
