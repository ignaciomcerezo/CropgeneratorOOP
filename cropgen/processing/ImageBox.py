from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from shapely import Polygon
from cropgen.processing.helpers.PairingErrors import (
    RepeatedSameAssociationError,
    MultipleAssociationError,
    NoAssociationError,
)
from cropgen.processing.helpers.helper_to_classes import (
    get_rotated_region,
    unrotate_image,
)
from shapely import Polygon, box as boxshape


@dataclass(slots=True, kw_only=True)
class ImageBox:
    """
    Contenedor de la información sobre las selecciones en la imagen hechas durante las anotaciones. Contiene información
    sobre el polígono dibujado, la rotación del polígono
    """

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

    def associate_fragment(self, fragment: "TextFragment", warn: bool = False):
        """
        Asocia un fragmento a nuestra caja-imagen. Si ya tiene uno asociado, salta un error.
        """
        if (
            len(self.associated_fragments) != 0
        ):  # si ya tenemos un fragmento de texto asociado
            if warn and (fragment.row_id in self.associated_fragments):
                raise RepeatedSameAssociationError(self)
            elif warn:
                raise MultipleAssociationError(self)

        self.associated_fragments.append(fragment)

        # self.corrected_centroid = None ??? why was this here #TODO: check why this was here
        # self.corrected_polygon = None

    def __hash__(self):
        return (
            self.id.__hash__()
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra sino de external_interfaces

    def __repr__(self):
        return (
            "<ImageBox "
            + ("rectangular" if self.true_rectangle else "poligonal")
            + f" {self.id} de la tarea ({self.task_id})."
            + self.unrotated * "¡Rotación cancelada!"
            + ">"
        )

    @property
    def fragment(self) -> "TextFragment":
        """Devuelve el primer fragmento de los asociados. Si hay más de uno, salta un error."""
        if len(self.associated_fragments) == 0:
            raise NoAssociationError(self)
        else:
            return self.associated_fragments[0]

    def centroid(self) -> tuple[float, float]:
        """Devuelve el centroide del pológono asociado a esta caja-imagen."""
        pol_centroid = self.polygon.centroid
        return pol_centroid.x, pol_centroid.y

    @property
    def top(self):
        """Coordenada y menor del polígono asociado."""
        return self.polygon.bounds[1]

    @property
    def left(self):
        """Coordenada x menor del polígono asociado."""
        return self.polygon.bounds[0]

    @staticmethod
    def from_json_value(
        json_value: dict,
        imgbox_id: str,
        task_id: int | str,
        img: Image.Image,
        unrotate: bool = False,
    ) -> "ImageBox":
        return ImageBox(
            id=imgbox_id,
            task_id=task_id,
            **ImageBox._rotatedregion(img, json_value, unrotate),
        )

    @staticmethod
    def _rotatedregion(
        img: Image.Image, json_value, unrotate=False
    ) -> dict[str, Image.Image | Polygon | bool]:
        """
        Genera parte de la información necesaria para instanciar ImageBox a partir de la información en una tarea y
        su imagen. Devuelve, en orden:
            crop: PIL.Image.Image -> imagen recortada
            polygon: Polygon -> polígono de shapely
            rotation: bool -> ángulo de rotación (manual o calculado) de la región para su lectura.
            true_rectangle: bool -> si se generó usando la herramienta rectángulo.

        Si unrotate=True, la imagen se endereza y el polígono se reconstruye para coincidir exactamente con
        las nuevas dimensiones visuales. El enderezado se hace caja por caja, por lo que no se preservan adyacencias y
        la imagen completa, si se representa con los objetos enderezados, puede presentar anomalías: el enderezado
         solamente es para el proceso de revisión,
        """
        height = json_value["original_height"]
        width = json_value["original_width"]
        val = json_value["value"]

        # Obtenemos el recorte y el polígono original (en coordenadas globales)
        crop, original_poly, rotation, polygonic = get_rotated_region(
            val, width, height, img
        )

        if not unrotate or not rotation:
            return {
                "crop": crop,
                "polygon": original_poly,
                "rotation": rotation,
                "true_rectangle": not polygonic,
                "unrotated": False,
            }
        else:
            # Si des-rotamos, la bounding box del polígono original (rotado) suele ser
            # más grande que la imagen enderezada final, generando espacios en blanco en el collage.
            # Procedemos a crear un nuevo polígono ajustado al píxel.

            # Generamos primero la imagen final para obtener sus dimensiones reales (w, h)
            # Esto elimina las zonas transparentes sobrantes tras la rotación.
            final_crop = unrotate_image(crop, rotation)
            cw, ch = final_crop.size

            # calculamos el punto de anclado basándonos en la geometría original.
            # Usamos el mínimo rectángulo rotado para hallar la verdadera esquina
            # superior izquierda visual, independientemente de la orientación de los ejes.
            rotated_rect = original_poly.minimum_rotated_rectangle
            rect_coords = list(rotated_rect.exterior.coords)[:-1]

            # ordenamos vértices: prioridad menor Y (arriba), desempate menor X (izquierda) - extremadamente improbable salvo en el borde inferior
            pivot_point = sorted(rect_coords, key=lambda p: (p[1], p[0]))[0]
            pivot_x, pivot_y = pivot_point

            # construimos un nuevo polígono rectangular
            # empieza en el pivote original pero tiene exactamente las dimensiones de la imagen recortada.
            # esto nos asegura consistencia al pegar en el lienzo del collage.
            new_poly = boxshape(pivot_x, pivot_y, pivot_x + cw, pivot_y + ch)

            return {
                "crop": final_crop,
                "polygon": new_poly,
                "rotation": rotation,
                "true_rectangle": not polygonic,
                "unrotated": True,
            }
