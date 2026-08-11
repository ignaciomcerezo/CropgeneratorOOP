from multiprocessing import Value
from cropgen.ocrdataset.layout_generator.transforms import (
    IntraparagraphTransform,
    InterparagraphTransform,
)
from cropgen.processing.AnnotatedPage import AnnotatedPage
from typing import Collection, Literal, Any, Sequence, Sequence, Optional
from torch.utils.data import Dataset
import numpy as np


class OCRDataset(Dataset):
    """
    Dataset variant intended to be used for OCR tasks. Takes as input a sequence of annotations
    (of type AnnotatedPage) and a collecion of orders that will be used to sample the pages.

    When an item is solicited, the dataset deterministically chooses an item (taken from all possible contiguous
    clusters of lines of one of the orders provided) and returns the crop, its transcription, and other data.
    """

    def __init__(
        self,
        annotations: Sequence[AnnotatedPage],
        *,
        orders: Collection[int | Literal["paragraph", "full"]],
        intraparagraph_transforms: Collection[IntraparagraphTransform],
        interparagraph_transforms: Collection[InterparagraphTransform],
    ):
        self.annotated_pages = annotations
        # temp
        self._orders = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self._update_orders(orders)  # the three previous attributes are updated here

        self._intraparagraph_transforms = intraparagraph_transforms
        self._interparagraph_transforms = interparagraph_transforms

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, value):
        self._update_orders(value)

    def _update_orders(
        self,
        new_orders: Collection[int | Literal["paragraph", "full"]],
    ):
        for order in new_orders:
            if (
                (isinstance(order, float))
                or (
                    isinstance(order, str)
                    and ((order != "full") and (order != "paragraph"))
                )
                or (isinstance(order, int) and (order < 1))
            ):
                raise ValueError(
                    f"Value {order} found inside orders. Only ints > 1, 'paragraph' and 'full' are acceptable orders."
                )

        pseudo_old_orders = set(self._orders)

        if self._use_paragraphs:
            pseudo_old_orders.add("paragraph")
        if self._use_full_pages:
            pseudo_old_orders.add("full")

        if set(new_orders) != pseudo_old_orders:
            self._orders = [x for x in new_orders if isinstance(x, int)]
            self._use_paragraphs = "paragraphs" in new_orders
            self._use_full_pages = "full" in new_orders
            self._recalculate_size_and_probabilities()

    def _update_transforms(
        self,
        intraparagraph_transforms: Collection[IntraparagraphTransform],
        interparagraph_transforms: Collection[InterparagraphTransform],
    ):
        self._intraparagraph_transforms = intraparagraph_transforms
        self._interparagraph_transforms = interparagraph_transforms

    def update_stage(
        self,
        new_orders: list[int],
        intraparagraph_transforms: Collection[IntraparagraphTransform],
        interparagraph_transforms: Collection[InterparagraphTransform],
    ):
        self._update_orders(new_orders)
        self._update_transforms(intraparagraph_transforms, interparagraph_transforms)

    def _recalculate_size_and_probabilities(self):
        page_sample_counts: list[int] = []
        par_prefix_sums: list[np.ndarray] = []
        page_prefix_sums: list[int] = []
        running_total: int = 0

        for ann in self.annotated_pages:
            par_counts: list[int] = []

            for par in ann.paragraphs:
                par_len = len(par)

                # Number of valid sliding windows of order k
                ell = sum(max(0, par_len - order + 1) for order in self._orders) + int(
                    self._use_paragraphs
                )

                par_counts.append(ell)

            page_par_samples = sum(par_counts)
            page_total_samples = page_par_samples + int(self._use_full_pages)

            par_prefix_sums.append(np.cumsum(par_counts, dtype=np.int64))
            page_sample_counts.append(page_total_samples)

            running_total += page_total_samples
            page_prefix_sums.append(running_total)

        self._size = running_total
        self._page_sample_counts = page_sample_counts
        self._page_prefix_sums = np.array(page_prefix_sums, dtype=np.int64)
        self._par_prefix_sums = par_prefix_sums

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Chooses a line cluster / paragraph / full page according to the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        page_idx = int(np.searchsorted(self._page_prefix_sums, index, side="right"))
        page_offset = int(self._page_prefix_sums[page_idx - 1]) if page_idx > 0 else 0
        rel_idx = index - page_offset

        ann: AnnotatedPage = self.annotated_pages[page_idx]

        if self._use_full_pages and rel_idx == self._page_sample_counts[page_idx] - 1:
            selected_box_ids = list(ann.image_boxes.keys())
            order = "full"
            identifyer = f"pg{page_idx}"
        else:
            par_prefix = self._par_prefix_sums[page_idx]
            par_idx = int(np.searchsorted(par_prefix, rel_idx, side="right"))
            par_offset = int(par_prefix[par_idx - 1]) if par_idx > 0 else 0
            item_idx = rel_idx - par_offset

            paragraph = ann.paragraphs[par_idx]
            line_ids = paragraph.image_boxes_ids
            num_lines = len(line_ids)

            curr = item_idx
            selected_box_ids: Optional[list[str]] = None

            for order in self._orders:
                n_windows = max(0, num_lines - order + 1)
                if curr < n_windows:
                    start_line = curr
                    selected_box_ids = line_ids[start_line : start_line + order]
                    identifyer = (
                        f"pg{page_idx}par{par_idx}L{start_line}"
                        + f"{start_line+order-1}" * (order > 1)
                    )
                    break
                curr -= n_windows

            if selected_box_ids is None and self._use_paragraphs:
                selected_box_ids = line_ids
                order = "paragraph"
                identifyer = f"pg{page_idx}par{par_idx}"

            if selected_box_ids is None:
                raise RuntimeError(f"Failed to resolve sample target for index {index}")

        # TODO: implement the layout changes.
        target_ann = ann

        collage, text, sindex = target_ann.cluster_reading_order(selected_box_ids)

        # TODO: add context?

        return {
            "image": collage,
            "text": text,
            "sindex": sindex,
            "order": order,
            "id": identifyer,
            "page_id": target_ann.task_id,
        }
