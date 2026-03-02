from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from shapely import Polygon
from display import display


class MultipleAssociationError(ValueError):
    pass


class RepeatedSameAssociationError(ValueError):
    pass


class NoAssociationError(ValueError):
    pass


@dataclass(slots=True, kw_only=True)
class ImageBox:
    id: str
    crop: Image.Image
    polygon: Polygon
    rotation: float
    unrotated: bool
    task_id: int
    associated_fragments: list["TextFragment"] = field(default_factory=lambda: list())
    true_rectangle: bool
    corrected_centroid: Optional[tuple[float, float]] = None
    corrected_polygon: Optional[Polygon] = None

    def associate_fragment(self, fragment: "TextFragment", warn: bool = True):
        if (
            len(self.associated_fragments) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if warn and (fragment.id in self.associated_fragments):
                display(self.crop)
                display(f"Fragmento: {fragment.text}")
                raise RepeatedSameAssociationError(
                    f"(Tarea {self.task_id}) - Asociación repetida: La imagen {self.id} tiene asociado el fragmento {fragment.id} más de una vez."
                )
            else:

                display(self.crop)
                display(f"Fragmento 1: {fragment.text}")
                for i, old_fragment in enumerate(self.associated_fragments):
                    display(f"Fragmento {i + 2}: {old_fragment.text}")
                raise MultipleAssociationError(
                    f"(Tarea {self.task_id}) - Multiasociación: La imagen {self.id} tiene asociados varios fragmentos:"
                )

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
            raise MultipleAssociationError(
                f"(Tarea {self.task_id}) La caja-imagen {self.id} de la tarea {self.task_id} no tiene fragmento de texto asociado."
            )
        elif len(self.associated_fragments) == 1:
            return self.associated_fragments[0]
        else:
            raise NoAssociationError(
                f"(Tarea {self.task_id}) La caja-imagen {self.id} de la tarea {self.task_id} tiene más de un fragmento asociado: {' '.join([f.text for f in self.associated_fragments])}"
            )

    def centroid(self) -> tuple[float, float]:
        pol_centroid = self.polygon.centroid
        return pol_centroid.x, pol_centroid.y

    @property
    def top(self):
        return self.polygon.bounds[1]

    @property
    def left(self):
        return self.polygon.bounds[0]
