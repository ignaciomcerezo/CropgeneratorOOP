import shapely
from shapely import Polygon
from cropgen.processing.line import Line
from cropgen.processing.paragraph import Paragraph
from typing import Collection, Sequence
from PIL import Image


def _union(geometries: Collection[shapely.Geometry]) -> Polygon:
    snapped_geoms = [shapely.set_precision(g, grid_size=1e-6) for g in geometries]
    return shapely.unary_union(snapped_geoms)


def _bounds(box: Line):
    return (box.left, box.top, box.right, box.bot)


class LineGroupInfo:
    """
    Some helpful data to calculate layouts.
    """

    def __init__(self, line_group: Paragraph | Sequence[Line]):

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
