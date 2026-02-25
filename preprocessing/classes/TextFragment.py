from dataclasses import dataclass, field
from preprocessing.classes.ImageBox import ImageBox
from display import display


@dataclass(slots=True, kw_only=True)
class TextFragment:
    id: str
    text: str
    task_id: int
    associated_boxes: list[ImageBox] = field(default_factory=lambda: list())
    starting_index: int = None

    def associate_box(self, box: ImageBox, warn: bool = True):

        if warn and (
            len(self.associated_boxes) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if box.id in self.associated_boxes:
                print(
                    f"(Tarea {self.task_id}) - Asociación repetida: El fragmento {self.id} tiene asociada la caja-imagen {box.id} más de una vez."
                )
                display(f"Fragmento: {self.text}")
                display(box.crop)
            else:
                print(
                    f"(Tarea {self.task_id}) - Multiasociación: El fragmento {self.id} tiene asociada más de una caja-imagen:"
                )
                display(f"Fragmento: {self.text}")
                display(box.crop)
                for old_box in self.associated_boxes:
                    display(old_box.crop)

        self.associated_boxes.append(box)

    def __hash__(self):
        return (
            self.id.__hash__()
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra, sino de label studio

    def __repr__(self):
        return f"TextFragment {self.id} de la tarea {self.task_id}"

    @property
    def box(self):
        """If the TextFragment has only one associated ImageBox, returns it.
        Ifit has more than one or none, raises a ValueError."""
        if len(self.associated_boxes) == 0:
            raise ValueError(
                f"El fragmento {self.id} de la tarea {self.task_id} no tiene caja-imagen asociada."
            )
        elif len(self.associated_boxes) == 1:
            return self.associated_boxes[0]
        else:
            raise ValueError(
                f"El fragmento {self.id} de la tarea {self.task_id} tiene más de una caja-imagen asociada: {' '.join(self.associated_fragments)}"
            )
