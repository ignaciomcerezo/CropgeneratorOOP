from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from shapely import Polygon
from preprocessing.helpers.PairingErrors import (
    RepeatedSameAssociationError,
    MultipleAssociationError,
    NoAssociationError,
)


@dataclass(slots=True, kw_only=True)
class ImageBox:
    id: str
    crop: Image.Image
    polygon: Polygon
    rotation: float
    unrotated: bool
    task_id: int
    index: Optional[int] = -1
    associated_fragments: list["TextFragment"] = field(default_factory=lambda: list())
    true_rectangle: bool
    corrected_centroid: Optional[tuple[float, float]] = None
    corrected_polygon: Optional[Polygon] = None

    def associate_fragment(self, fragment: "TextFragment", warn: bool = True):
        if (
            len(self.associated_fragments) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if warn and (fragment.id in self.associated_fragments):
                raise RepeatedSameAssociationError(self)
            else:
                raise MultipleAssociationError(self)

        self.associated_fragments.append(fragment)

        self.corrected_centroid = None
        self.corrected_polygon = None

    def __hash__(self):
        return (
            self.id.__hash__()
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra sino de labelstudio

    def __repr__(self):
        return (
            f"<ImageBox "
            + ("rectangular" if self.true_rectangle else "poligonal")
            + f" {self.id} de la tarea ({self.task_id})."
            + self.unrotated * "¡Rotación cancelada!"
            + ">"
        )

    @property
    def fragment(self) -> "TextFragment":
        """If the ImageBox has only one associated TextFragment, returns it.
        If it has more than one, raises a ValueError."""
        if len(self.associated_fragments) == 0:
            raise MultipleAssociationError(self)
        elif len(self.associated_fragments) != 1:
            raise NoAssociationError(self)
        else:
            return self.associated_fragments[0]

    def centroid(self) -> tuple[float, float]:
        pol_centroid = self.polygon.centroid
        return pol_centroid.x, pol_centroid.y

    @property
    def top(self):
        return self.polygon.bounds[1]

    @property
    def left(self):
        return self.polygon.bounds[0]
