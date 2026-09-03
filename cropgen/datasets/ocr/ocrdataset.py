from cropgen.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    _poss_cluster_args_literal,
    _default_cluster_param_values,
    _default_cluster_parameters,
    orders_type,
)
from cropgen.transforms.on_the_fly_transform_pack import OCROnTheFlyTransformPack
from cropgen.processing.annotated_page import AnnotatedPage
from typing import (
    Collection,
    Literal,
    Any,
    Sequence,
    Sequence,
    Optional,
    get_args,
    Callable,
)

_default_getitem_output_literal = Literal[
    "image",
    "text",
    "sindex",
    "context",
    "order",
    "id",
    "page_id",
]

_formatter_signature = Callable[[dict[_default_getitem_output_literal, Any]], Any]


class OCRDataset(BaseAnnotationDataset):
    """
    Dataset variant intended to be used for OCR tasks. Takes as input a sequence of annotations
    (of type AnnotatedPage) and a collecion of orders that will be used to sample the pages.

    When an item is requested, the dataset deterministically chooses an item (taken from all
    possible contiguous clusters of lines of length one of the orders provided), transforms it using
    the transform given (via .set_transform()) and returns the crop, its transcription, and other data.

    The output may be formatted using .set_formatter().
    """

    def __init__(
        self,
        annotations: Sequence[AnnotatedPage],
        *,
        orders: orders_type,
        cluster_transform_params: dict[_poss_cluster_args_literal, Any] | None = None,
    ):
        self._annotated_pages = annotations
        # temp
        self._orders: list[int] = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self._transforms: OCROnTheFlyTransformPack | None = None
        self._update_orders(orders)  # the three previous attributes are updated here
        self._formatter: _formatter_signature | None = None

        self._cluster_params = _default_cluster_parameters.copy()
        self._cluster_params.update(
            cluster_transform_params if cluster_transform_params is not None else dict()
        )

    def __repr__(self):
        return f"<OCRDataset ({len(self)} samples: {len(self._annotated_pages)} pages using orders {self.orders}>"

    def __getitem__(self, index: int):
        """
        Chooses a line cluster / paragraph / full page according to the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        ann, selected_line_ids, order, identifier = (
            self._gets_ann_ids_order_and_identifier(index)
        )

        synthetic_img, synthetic_transcription, sindex = ann.synthetic_sample(
            list(selected_line_ids),
            tight_layout=self._cluster_params["tight_layout"],
            margin_size_px=self._cluster_params["margin_size_px"],
            img_poly_transform=self._transforms,
            overlay_polygons=self._cluster_params["overlay_polygons"],
            overlay_mbr=self._cluster_params["overlay_mbr"],
        )

        # TODO: improve context generation - implement the use_previous_page_in_context cluster parameter here
        if sindex > 0:
            context = ann.full_transcription[:sindex]
        else:
            context = ""

        sample: dict[_default_getitem_output_literal, Any] = {
            "image": synthetic_img,
            "text": synthetic_transcription,
            "sindex": sindex,
            "context": context,
            "order": order,
            "id": identifier,
            "page_id": ann.task_id,
        }

        if self._formatter is None:
            return sample
        else:
            return self._formatter(sample)
