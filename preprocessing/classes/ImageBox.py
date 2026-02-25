from dataclasses import dataclass, field
from PIL.Image import Image
from shapely import Polygon
from display import display


@dataclass(slots=True, kw_only=True)
class ImageBox:
    id: str
    crop: Image
    polygon: Polygon
    rotation: float
    unrotated: bool
    task_id: int
    associated_fragments: list["TextFragment"] = field(default_factory=lambda: list())
    true_rectangle: bool

    def associate_fragment(self, fragment: "TextFragment", warn: bool = True):
        if (
            len(self.associated_fragments) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if warn and (fragment.id in self.associated_fragments):
                print(
                    f"(Tarea {self.task_id}) - Asociación repetida: La imagen {self.id} tiene asociado el fragmento {fragment.id} más de una vez."
                )
                display(self.crop)
                display(f"Fragmento: {fragment.text}")
            else:
                print(
                    f"(Tarea {self.task_id}) - Multiasociación: La imagen {self.id} tiene asociados varios fragmentos:"
                )
                display(self.crop)
                display(f"Fragmento 1: {fragment.text}")
                for i, old_fragment in enumerate(self.associated_fragments):
                    display(f"Fragmento {i + 2}: {old_fragment.text}")

        self.associated_fragments.append(fragment)

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
            raise ValueError(
                f"La caja-imagen {self.id} de la tarea {self.task_id} no tiene fragmento de texto asociado."
            )
        elif len(self.associated_fragments) == 1:
            return self.associated_fragments[0]
        else:
            raise ValueError(
                f"La caja-imagen {self.id} de la tarea {self.task_id} tiene más de un fragmento asociado: {' '.join(self.associated_fragments)}"
            )
