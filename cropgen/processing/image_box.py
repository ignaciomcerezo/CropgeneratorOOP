from cropgen.shared.LSTypedDicts.values import RectangleValue
from cropgen.shared.LSTypedDicts.values import PolygonValue
from dataclasses import dataclass, field
from typing import Optional
from typing import TYPE_CHECKING

from PIL import Image
from shapely import Polygon, box as boxshape
from shapely.affinity import scale

from cropgen.processing.helpers.PairingErrors import (
    RepeatedSameAssociationError,
    MultipleAssociationError,
    NoAssociationError,
)
from cropgen.processing.helpers.helper_to_classes import (
    get_rotated_region,
)
from cropgen.shared.LSTypedDicts.results import RectangleResult, PolygonResult

if TYPE_CHECKING:
    from cropgen.processing.text_fragment import TextFragment


@dataclass(slots=True, kw_only=True)
class ImageBox:
    """
    Contenedor de la información sobre las selecciones en la imagen hechas durante las anotaciones. Contiene información
    sobre el polígono dibujado, la rotación del polígono, el fragmentos asociado, el recorte correspondiente.
    """

    id: str
    # crop: Image.Image
    stroke_crop: Image.Image
    polygon: Polygon
    rotation: float
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
            if warn and (fragment.id in self.associated_fragments):
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

    @property
    def right(self):
        """Coordenada x mayor del polígono asociado."""
        return self.polygon.bounds[2]

    @property
    def bot(self):
        """Coordenada y mayor del polígono asociado."""
        return self.polygon.bounds[3]

    @staticmethod
    def from_image_result(
        simplified_result_item: RectangleResult | PolygonResult,
        task_id: int,
        stroke: Image.Image,
    ) -> "ImageBox":
        imgbox_id = simplified_result_item.id

        residual_crop, polygon, rotation, true_rectangle = ImageBox._rotatedregion(
            stroke, simplified_result_item
        )

        return ImageBox(
            id=imgbox_id,
            task_id=task_id,
            stroke_crop=residual_crop,
            polygon=polygon,
            rotation=rotation,
            true_rectangle=true_rectangle,
        )

    @staticmethod
    def _rotatedregion(
        residual: Image.Image,
        simplified_result_item: RectangleResult | PolygonResult,
    ) -> tuple[Image.Image, Polygon, float, bool]:

        val: RectangleValue | PolygonValue = simplified_result_item.value

        residual_crop, original_poly, rotation, polygonic = get_rotated_region(
            val,
            residual,
        )

        return residual_crop, original_poly, rotation, not polygonic
