from shapely.geometry import Polygon
from cropgen.processing.image_box import ImageBox
from typing import Collection, Sequence
from cropgen.processing import Paragraph
from abc import ABC, abstractmethod
import shapely
from PIL import Image


def _union(geometries: Collection[shapely.Geometry]) -> Polygon:
    snapped_geoms = [shapely.set_precision(g, grid_size=1e-6) for g in geometries]
    return shapely.unary_union(snapped_geoms)


def _bounds(box: ImageBox):
    return (box.left, box.top, box.right, box.bot)


class ParagraphInfo:
    """
    Some helpful data to calculate layouts.
    """

    def __init__(self, line_group: Paragraph | Sequence[ImageBox]):

        # x0,y0,xf,yf
        self.box_bounds: list[tuple[float, float, float, float]] = [
            _bounds(box) for box in line_group
        ]

        self.area: float = _union([box.polygon for box in line_group]).area

        self.avg_rotation = (
            1
            / len(line_group)
            * sum((box.rotation * box.polygon.area) for box in line_group)
        )

        if not self.box_bounds:
            self.union_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
            self.centroid: tuple[float, float] = (0.0, 0.0)
        else:
            min_x = min(b[0] for b in self.box_bounds)
            min_y = min(b[1] for b in self.box_bounds)
            max_x = max(b[2] for b in self.box_bounds)
            max_y = max(b[3] for b in self.box_bounds)
            self.union_bounds = (min_x, min_y, max_x, max_y)
            self.centroid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

        self.n_points = len(line_group)
        self.box_x_deltas = [
            self.box_bounds[i + 1][0] - self.box_bounds[i][0]
            for i in range(len(line_group) - 1)
        ]
        self.box_y_deltas = [
            self.box_bounds[i + 1][1] - self.box_bounds[i][1]
            for i in range(len(line_group) - 1)
        ]
        self.union_polygon = _union([box.polygon for box in line_group])

        self.center = ((self.x0 + self.xf) / 2, (self.y0 + self.yf) / 2)

    @property
    def x0(self) -> float:
        return self.union_bounds[0]

    @property
    def xf(self) -> float:
        return self.union_bounds[2]

    @property
    def y0(self) -> float:
        return self.union_bounds[1]

    @property
    def yf(self) -> float:
        return self.union_bounds[3]

    @property
    def x0s(self) -> list[float]:
        return [bound[0] for bound in self.box_bounds]

    @property
    def xfs(self) -> list[float]:
        return [bound[2] for bound in self.box_bounds]

    @property
    def y0s(self) -> list[float]:
        return [bound[1] for bound in self.box_bounds]

    @property
    def yfs(self) -> list[float]:
        return [bound[3] for bound in self.box_bounds]

    @property
    def w(self) -> float:
        return abs(self.xf - self.x0)

    @property
    def h(self) -> float:
        return abs(self.yf - self.y0)

    @property
    def ws(self) -> list[float]:
        return [abs(x0 - xf) for (x0, xf) in zip(self.x0s, self.xfs)]

    @property
    def hs(self) -> list[float]:
        return [abs(y0 - yf) for (y0, yf) in zip(self.y0s, self.yfs)]


class LinewiseTransform(ABC):
    """
    Base class used to modify single linges, for example single line
    distortions and stretching.
    """

    @abstractmethod
    def __call__(self, box: ImageBox) -> tuple[Image.Image, shapely.Polygon]:
        raise NotImplementedError

    def bulk_transform(
        self, line_group: Paragraph | Sequence[ImageBox]
    ) -> tuple[list[Image.Image], list[Polygon]]:
        new_imgs, new_polygons = [], []

        for box in line_group:
            new_img, new_polygon = self(box)
            new_imgs.append(new_img)
            new_polygons.append(new_polygon)

        return new_imgs, new_polygons


class IntraparagraphTransform(ABC):
    """
    Base class used to modify layouts for individual paragraphs.
    For example line shears or paragraph rotations. or line-by-line distortions could be
    implemented like this.
    """

    @abstractmethod
    def __call__(
        self, line_group: Paragraph | Sequence[ImageBox]
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        raise NotImplementedError

    @staticmethod
    def from_linewise(transform: LinewiseTransform):
        return IntraparagraphFromLinewiseTransform(transform)


class IntraparagraphFromLinewiseTransform(IntraparagraphTransform):
    """
    Linewise transform turned paragraph transform.
    """

    def __init__(self, transform: LinewiseTransform):
        self._transform = transform

    def __call__(
        self, line_group: Paragraph | Sequence[ImageBox]
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        return self._transform.bulk_transform(line_group)


class InterparagraphTransform(ABC):
    """
    Base class used to modify layouts for complete documents.
    For example this could be used to separate paragraphs between them,
    rotate them globally, etc.
    """

    @abstractmethod
    def __call__(self, *line_groups_group: Paragraph | Sequence[ImageBox]) -> None:
        raise NotImplementedError
