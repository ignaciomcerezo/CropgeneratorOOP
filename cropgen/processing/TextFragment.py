from dataclasses import dataclass, field
from typing import Optional
from cropgen.processing.ImageBox import (
    ImageBox,
)
from cropgen.processing.helpers.PairingErrors import (
    RepeatedSameAssociationError,
    MultipleAssociationError,
    NoAssociationError,
)

_things_to_close = (
    ("{", "}"),
    ("$", "$"),
)


@dataclass(slots=True, kw_only=True)
class TextFragment:
    id: str
    text: str
    task_id: int
    associated_boxes: list[ImageBox] = field(default_factory=list)
    starting_index: Optional[int] = None

    def associate_box(self, box: ImageBox, warn: bool = False):

        if warn and (
            len(self.associated_boxes) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if box.id in self.associated_boxes:
                raise RepeatedSameAssociationError(self)
            elif warn:
                raise MultipleAssociationError(self)

        self.associated_boxes.append(box)

    def __hash__(self):
        return (
            self.id.__hash__()
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra, sino de label studio

    def __repr__(self):
        return f"<TextFragment {self.id} de la tarea ({self.task_id})>."

    @property
    def box(self):
        """Returns the first of the associated boxes of the text fragment."""
        if len(self.associated_boxes) == 0:
            raise NoAssociationError(self)
        else:
            return self.associated_boxes[0]
