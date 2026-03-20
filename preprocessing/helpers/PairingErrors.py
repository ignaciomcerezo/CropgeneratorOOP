class PairingError(ValueError):
    pass


class RepeatedSameAssociationError(PairingError):
    def __init__(self, box_or_fragment):
        object_type = type(box_or_fragment).__name__
        if object_type == "ImageBox":
            msg = f"(Tarea {box_or_fragment.task_id}) - Asociación repetida: La caja-imagen {box_or_fragment.id} tiene asociado un mismo fragmento más de una vez:"
            for i, fragment in enumerate(box_or_fragment.associated_fragments):
                msg += f"\n\tAsociación (fragmento) {i}: {fragment.text}."
            self.message = msg
        elif object_type == "TextFragment":
            msg = f"(Tarea {box_or_fragment.task_id}) - Asociación repetida: El fragmento de texto {box_or_fragment.id} tiene asociada una misma caja-imagen más de una vez:"
            for i, box in enumerate(box_or_fragment.associated_boxes):
                msg += f"\n\tAsociación (caja) {i}: {box.id}."
            self.message = msg
        else:
            raise ValueError(
                f"No se ha detectado que sea ni un ImageBox ni un TextFragment, el tipo {object_type} no se acepta."
            )
        super().__init__(self.message)


class MultipleAssociationError(PairingError):
    def __init__(self, box_or_fragment):
        object_type = type(box_or_fragment).__name__

        if object_type == "ImageBox":
            msg = f"(Tarea {box_or_fragment.task_id}) - Multiasociación: La caja-imagen {box_or_fragment.id} tiene asociado más de un fragmento de texto:"
            for i, fragment in enumerate(box_or_fragment.associated_fragments):
                msg += f"\n\tAsociación (fragmento) {i}: {fragment.text}"
            self.message = msg

        elif object_type == "TextFragment":
            msg = f"(Tarea {box_or_fragment.task_id}) - Multiasociación: El fragmento de texto {box_or_fragment.id} tiene asociada más de una caja-imagen:"
            for i, box in enumerate(box_or_fragment.associated_boxes):
                msg += f"\n\tAsociación (caja) {i}: {box.id}"
            self.message = msg
        else:
            raise ValueError(
                f"No se ha detectado que sea ni un ImageBox ni un TextFragment, el tipo {object_type} no se acepta."
            )
        super().__init__(self.message)


class NoAssociationError(PairingError):
    def __init__(self, box_or_fragment):
        object_type = type(box_or_fragment).__name__
        if object_type == "ImageBox":
            self.message = f"(Tarea {box_or_fragment.task_id}) - Sin asociación: La caja-imagen {box_or_fragment.id} no tiene asociado ningún fragmento de texto."

        elif object_type == "TextFragment":
            self.message = f"(Tarea {box_or_fragment.task_id}) - Sin asociación: El fragmento de texto {box_or_fragment.id} no tiene asociada ninguna caja-imagen."
        else:
            raise ValueError(
                f"No se ha detectado que sea ni un ImageBox ni un TextFragment, el tipo {object_type} no se acepta."
            )
        super().__init__(self.message)


class SameToSameAssociation(PairingError):
    def __init__(self, box_or_fragment):
        object_type = type(box_or_fragment).__name__
        if object_type == "ImageBox":

            erroneous_links = [
                obj
                for obj in box_or_fragment.associated_fragments
                if type(obj).__name__ == "ImageBox"
            ]
            msg = f"(Tarea {box_or_fragment.task_id}) - Asociación entre elementos del mismo tipo: La caja-imagen {box_or_fragment.id} tiene asociadas cajas-imagen:"
            for obj in erroneous_links:
                msg += f"\n\tCaja {obj.id}"

            self.message = msg

        elif object_type == "TextFragment":
            erroneous_links = [
                obj
                for obj in box_or_fragment.associated_boxes
                if type(obj).__name__ == "TextFragment"
            ]
            msg = f"(Tarea {box_or_fragment.task_id}) - Asociación entre elementos del mismo tipo: El fragmento {box_or_fragment.id} tiene asociados fragmentos:"
            for obj in erroneous_links:
                msg += f"\n\tFragmento {obj.id}"

            self.message = msg
        else:
            raise ValueError(
                f"No se ha detectado que sea ni un ImageBox ni un TextFragment, el tipo {object_type} no se acepta."
            )
        super().__init__(self.message)
