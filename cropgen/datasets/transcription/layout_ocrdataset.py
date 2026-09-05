from copy import deepcopy
from torch.utils.data import Dataset
from cropgen.datasets.helpers.layout_generator import LayoutGenerator
from typing import Sequence, Any, Literal
from cropgen.ocr_units import OCRPage
from cropgen.datasets.base_annotation_dataset import (
    orders_type,
    _poss_cluster_args_literal,
    _default_cluster_parameters,
)
from cropgen.datasets.ocr_transform_pack import OCRTransformPack
from cropgen.datasets.transcription.ocrdataset import OCRDataset


class LayoutOCRDataset(Dataset):
    """
    Dataset variant intended to be used for OCR model training. It is built atop
    cropgen.datasets.OCRDataset, but implements more agressive layout modification:
    When .refresh_layouts() is called, the base OCRDataset is copied and each page´
    modified, changing the layout (via InterparagraphTransform).
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        layout_generator: LayoutGenerator,
        *,
        orders: orders_type,
        cluster_transform_params: dict[_poss_cluster_args_literal, Any] | None = None,
    ):
        self._layout_generator = deepcopy(layout_generator)
        self._base_annotations = annotations
        self._set_underlying(orders=orders, params=cluster_transform_params)

    def _set_underlying(
        self,
        orders: orders_type,
        params: dict[_poss_cluster_args_literal, Any] | None = None,
    ):
        new_anns = []
        for ann in self._base_annotations:
            new_anns.append(self._layout_generator.apply(ann))
        self._underlying_dataset = OCRDataset(
            annotations=new_anns,
            orders=orders,
            cluster_transform_params=params,
        )

    def refresh_layouts(self):
        new_anns = []
        for ann in self._base_annotations:
            new_anns.append(self._layout_generator.apply(ann))
        self._underlying_dataset = OCRDataset(
            annotations=new_anns,
            orders=self.orders,
            cluster_transform_params=self.cluster_params,
        )

    @property
    def orders(self):
        return self._underlying_dataset.orders

    @orders.setter
    def orders(self, value: Sequence[int | Literal["paragraph", "page"]]):

        self._underlying_dataset.orders = value

    @property
    def cluster_params(self):
        return self._underlying_dataset._cluster_params

    def set_cluster_param(
        self, cluster_param_name: _poss_cluster_args_literal, value: Any
    ):
        self._underlying_dataset.set_cluster_param(cluster_param_name, value)

    @property
    def layout_generator(self) -> LayoutGenerator:
        return self._layout_generator

    @layout_generator.setter
    def layout_generator(self, value: LayoutGenerator):
        self._layout_generator = value
        self.refresh_layouts()

    def __len__(self):
        return len(self._underlying_dataset)

    def __getitem__(self, index):
        return self._underlying_dataset[index]
