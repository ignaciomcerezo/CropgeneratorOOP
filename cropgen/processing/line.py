from cropgen.shared.LSTypedDicts.simplified import SimplifiedTextCorrectionResult
from cropgen.shared.LSTypedDicts.values import RectangleValue
from cropgen.shared.LSTypedDicts.values import PolygonValue
from dataclasses import dataclass
from typing import Optional

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


@dataclass(slots=True, kw_only=True)
class Line:
    """
    Contenedor de la información sobre las selecciones en la imagen hechas durante las anotaciones. Contiene información
    sobre el polígono dibujado, la rotación del polígono, el fragmentos asociado, el recorte correspondiente.
    """

    box_id: str
    fragment_id: str
    # crop: Image.Image
    stroke_crop: Image.Image
    polygon: Polygon
    rotation: float
    task_id: int
    text: str
    index: Optional[int] = -1
    true_rectangle: bool
    corrected_centroid: Optional[tuple[float, float]] = None
    starting_index: Optional[int] = None

    @property
    def id(self) -> str:
        return self.box_id + "-@-" + self.fragment_id

    def __hash__(self):
        return hash(
            str(self.box_id) + str(self.fragment_id)
        )  # podemos devolver el id sabiendo que, en caso de colisión, no es culpa nuestra sino de external_interfaces

    def __repr__(self):
        return f"<Line with box {self.box_id} and fragment {self.fragment_id} of task {self.task_id}>"

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
    def from_matching_ann_results(
        simplified_img_result_item: RectangleResult | PolygonResult,
        simplified_txt_result_item: SimplifiedTextCorrectionResult,
        task_id: int,
        stroke: Image.Image,
    ) -> "Line":
        imgbox_id = simplified_img_result_item.id

        residual_crop, polygon, rotation, true_rectangle = Line._rotatedregion(
            stroke, simplified_img_result_item
        )

        return Line(
            box_id=imgbox_id,
            task_id=task_id,
            stroke_crop=residual_crop,
            polygon=polygon,
            rotation=rotation,
            true_rectangle=true_rectangle,
            fragment_id=simplified_txt_result_item.id,
            text=" ".join(simplified_txt_result_item.value.text).strip(),
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
