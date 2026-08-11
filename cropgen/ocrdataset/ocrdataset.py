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
        self.__orders = []
        self.__update_orders(orders)

        self.__intraparagraph_transforms = intraparagraph_transforms
        self.__interparagraph_transforms = interparagraph_transforms

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, value):
        self.__update_orders(value)

    def __update_orders(
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

        if new_orders != self.__orders:
            self.__orders = [x for x in new_orders if isinstance(x, int)]
            self.__use_paragraphs = "paragraphs" in new_orders
            self.__use_full_pages = "full" in new_orders
            self.__recalculate_size_and_probabilities()

    def __update_transforms(
        self,
        intraparagraph_transforms: Collection[IntraparagraphTransform],
        interparagraph_transforms: Collection[InterparagraphTransform],
    ):
        self.__intraparagraph_transforms = intraparagraph_transforms
        self.__interparagraph_transforms = interparagraph_transforms

    def update_stage(
        self,
        new_orders: list[int],
        intraparagraph_transforms: Collection[IntraparagraphTransform],
        interparagraph_transforms: Collection[InterparagraphTransform],
    ):
        self.__update_orders(new_orders)
        self.__update_transforms(intraparagraph_transforms, interparagraph_transforms)

    def __recalculate_size_and_probabilities(self):
        page_sample_counts: list[int] = []
        par_prefix_sums: list[np.ndarray] = []
        page_prefix_sums: list[int] = []
        running_total: int = 0

        for ann in self.annotated_pages:
            par_counts: list[int] = []

            for par in ann.paragraphs:
                par_len = len(par)

                # Number of valid sliding windows of order k
                ell = sum(max(0, par_len - order + 1) for order in self.__orders) + int(
                    self.__use_paragraphs
                )

                par_counts.append(ell)

            page_par_samples = sum(par_counts)
            page_total_samples = page_par_samples + int(self.__use_full_pages)

            par_prefix_sums.append(np.cumsum(par_counts, dtype=np.int64))
            page_sample_counts.append(page_total_samples)

            running_total += page_total_samples
            page_prefix_sums.append(running_total)

        self.__size = running_total
        self.__page_sample_counts = page_sample_counts
        self.__page_prefix_sums = np.array(page_prefix_sums, dtype=np.int64)
        self.__par_prefix_sums = par_prefix_sums

    def __len__(self) -> int:
        return self.__size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Chooses a line cluster / paragraph / full page according to the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        """
        if index < 0 or index >= self.__size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self.__size}"
            )

        page_idx = int(np.searchsorted(self.__page_prefix_sums, index, side="right"))
        page_offset = int(self.__page_prefix_sums[page_idx - 1]) if page_idx > 0 else 0
        rel_idx = index - page_offset

        ann: AnnotatedPage = self.annotated_pages[page_idx]

        if self.__use_full_pages and rel_idx == self.__page_sample_counts[page_idx] - 1:
            selected_box_ids = list(ann.image_boxes.keys())
            order = "full"
            identifyer = page_idx
        else:
            par_prefix = self.__par_prefix_sums[page_idx]
            par_idx = int(np.searchsorted(par_prefix, rel_idx, side="right"))
            par_offset = int(par_prefix[par_idx - 1]) if par_idx > 0 else 0
            item_idx = rel_idx - par_offset

            paragraph = ann.paragraphs[par_idx]
            line_ids = paragraph.image_boxes_ids
            num_lines = len(line_ids)

            curr = item_idx
            selected_box_ids: Optional[list[str]] = None

            for order in self.__orders:
                n_windows = max(0, num_lines - order + 1)
                if curr < n_windows:
                    start_line = curr
                    selected_box_ids = line_ids[start_line : start_line + order]
                    identifyer = f"L{start_line}" + f"{start_line+order-1}" * (
                        order > 1
                    )
                    break
                curr -= n_windows

            if selected_box_ids is None and self.__use_paragraphs:
                selected_box_ids = line_ids
                order = "paragraph"
                identifyer = f"P{par_idx}"

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
            "box_ids": selected_box_ids,
            "page_id": target_ann.task_id,
        }
