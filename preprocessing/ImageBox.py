from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from shapely import Polygon
from display import display


class PairingError(ValueError):
    pass


class RepeatedSameAssociationError(PairingError):

    pass


class MultipleAssociationError(PairingError):
    def __init__(self, box_or_fragment):
        object_type = type(box_or_fragment).__name__
        if object_type == "ImageBox":
            msg = f"(Task {box_or_fragment.task_id}) - Multiasociación: La caja-imagen {box_or_fragment.id} tiene asociado más de un fragmento de texto:"
            for i, fragment in enumerate(box_or_fragment.associated_fragments):
                msg += f"Fragmento {i}: {fragment.text}"
            self.message = msg
        elif object_type == "TextFragment":
            msg = f"(Task {box_or_fragment.task_id}) - Multiasociación: El fragmento de texto {box_or_fragment.id} tiene asociada más de una caja-imagen:"
            for i, box in enumerate(box_or_fragment.associated_boxes):
                msg += f"Caja {i}: {box.id}"
            self.message = msg
        else:
            raise ValueError(
                f"No se ha detectado que sea ni un ImageBox ni un TextFragment, el tipo {object_type} no se acepta."
            )



class NoAssociationError(PairingError):
    pass


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
                display(self.crop)
                display(f"Fragmento: {fragment.text}")
                raise RepeatedSameAssociationError(
                    f"(Tarea {self.task_id}) - Asociación repetida: La imagen {self.id} tiene asociado el fragmento {fragment.id}, texto {fragment.text} más de una vez."
                )
            else:
                fragmentos_string = f"\nFragmento 1: {fragment.text}"
                for i, old_fragment in enumerate(self.associated_fragments):
                    fragmentos_string += f"\nFragmento {i + 2}: {old_fragment.text}"

                display(self.crop)
                raise MultipleAssociationError(
                    f"(Tarea {self.task_id}) - Multiasociación: La imagen {self.id} tiene asociados varios fragmentos: {fragmentos_string}"
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
        elif len(self.associated_fragments) != 1:
            raise NoAssociationError(
                f"(Tarea {self.task_id}) La caja-imagen {self.id} de la tarea {self.task_id} tiene más de un fragmento asociado: {' '.join([f.text for f in self.associated_fragments])}"
            )
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
