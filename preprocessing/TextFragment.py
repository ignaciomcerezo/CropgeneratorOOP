from dataclasses import dataclass, field
import re
from typing import Optional
from preprocessing.ImageBox import (
    ImageBox,
    NoAssociationError,
    MultipleAssociationError,
)
from display import display

_things_to_close = (
    ("{", "}"),
    ("$", "$"),
)


@dataclass(slots=True, kw_only=True)
class TextFragment:
    id: str
    text: str
    task_id: int
    associated_boxes: list[ImageBox] = list
    starting_index: Optional[int] = None

    math_percentage: float = field(init=False)
    is_open: bool = field(init=False)
    word_count: int = field(init=False)
    char_count: int = field(init=False)

    def __post_init__(self):
        self.math_percentage = self._math_percentage()
        self.is_open = self._is_open()
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)

    def associate_box(self, box: ImageBox, warn: bool = True):

        if warn and (
            len(self.associated_boxes) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if box.id in self.associated_boxes:
                display(f"Fragmento: {self.text}")
                display(box.crop)
                raise MultipleAssociationError(
                    f"(Tarea {self.task_id}) El fragmento {self.id} tiene asociada la caja-imagen {box.id} más de una vez."
                )
            else:
                display(f"Fragmento: {self.text}")
                display(box.crop)
                for old_box in self.associated_boxes:
                    display(old_box.crop)
                raise MultipleAssociationError(
                    f"(Tarea {self.task_id}) - Multiasociación: El fragmento {self.id} tiene asociada más de una caja-imagen:"
                )

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
            boxes_ids = [box.id for box in self.associated_boxes]
            raise ValueError(
                f"El fragmento {self.id} de la tarea {self.task_id} tiene más de una caja-imagen asociada: {' '.join(boxes_ids)}"
            )

    def _is_open(
        self, thigs_to_close=_things_to_close
    ) -> bool:  # TODO: import this kind of checks

        for opener, closer in _things_to_close:
            if opener != closer and (
                self.text.count(opener) != self.text.count(closer)
            ):
                return True
            elif opener == closer and (self.text.count(opener) % 2):
                return True
        return False

    def text_inside_math(self):
        return extract_math_from_dollars(self.text)

    def text_outside_math(self):
        extract_math_from_dollars("$" + self.text + "$")

    def _math_percentage(self):
        if self._is_open():
            return -1
        in_math_length = sum([len(block) for block in self.text_inside_math()])
        out_math_length = sum([len(block) for block in self.text_outside_math()])
        return in_math_length / (out_math_length + in_math_length)


def extract_math_from_dollars(text):
    pattern = r"(?<!\\)(\$\$?)(.*?)(?<!\\)\1"

    matches = re.findall(pattern, text, flags=re.DOTALL)
    extracted_text = [match[1].strip() for match in matches]

    return extracted_text
