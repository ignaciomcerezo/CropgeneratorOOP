from dataclasses import dataclass
from typing import Optional

from PIL import Image
from shapely import Polygon, box as boxshape
from shapely.affinity import scale


@dataclass(slots=True, kw_only=True)
class Line:
    """
    Contains teh information about a single line: the polygon it occupies in the page, its stroke crop and its transcription.
    """

    id: str
    stroke_crop: Image.Image
    polygon: Polygon
    rotation: float
    task_id: int
    text: str
    index: Optional[int] = -1
    corrected_centroid: Optional[tuple[float, float]] = None
    starting_index: Optional[int] = None

    def __hash__(self):
        return hash(
            self.id
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra sino de external_interfaces

    def __repr__(self):
        return f"<Line with id {self.id} of task {self.task_id}>"

    def centroid(self) -> tuple[float, float]:
        """Centroid of the associated polygon."""
        pol_centroid = self.polygon.centroid
        return pol_centroid.x, pol_centroid.y

    @property
    def top(self):
        """Lowest y coordinate (documents usually are y-down)."""
        return self.polygon.bounds[1]

    @property
    def left(self):
        """Lowest x coordinate."""
        return self.polygon.bounds[0]

    @property
    def right(self):
        """Greatest x coordinate."""
        return self.polygon.bounds[2]

    @property
    def bot(self):
        """Greatest y coordinate (documents usually are y-down)."""
        return self.polygon.bounds[3]
