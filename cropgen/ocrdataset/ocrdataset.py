from cropgen.transforms.on_the_fly_transform_manager import OCROnTheFlyTransformPack
from dataclasses import dataclass
from multiprocessing import Value
from cropgen.transforms import (
    IntraparagraphTransform,
    InterparagraphTransform,
)
from cropgen.processing.annotated_page import AnnotatedPage
from typing import Collection, Literal, Any, Sequence, Sequence, Optional
from torch.utils.data import Dataset
import numpy as np

orders_type = Collection[int | Literal["paragraph", "full"]]


_possible_cluster_args = [
    "tight_layout",
    "margin_size_px",
    "use_previous_page_in_context",
]
_poss_cluster_args_literal = Literal[
    "tight_layout", "margin_size_px", "use_previous_page_in_context"
]

_default_cluster_parameters: dict[_poss_cluster_args_literal, Any] = {
    "tight_layout": True,
    "margin_size_px": 0,
    "use_previous_page_in_context": False,
}


class OCRDataset(Dataset):
    """
    Dataset variant intended to be used for OCR tasks. Takes as input a sequence of annotations
    (of type AnnotatedPage) and a collecion of orders that will be used to sample the pages.

    When an item is requested, the dataset deterministically chooses an item (taken from all
    possible contiguous clusters of lines of length one of the orders provided) and returns the
    crop, its transcription, and other data.
    """

    def __init__(
        self,
        annotations: Sequence[AnnotatedPage],
        *,
        orders: orders_type,
        intraparagraph_transforms: Collection[IntraparagraphTransform] | None = None,
        interparagraph_transforms: Collection[InterparagraphTransform] | None = None,
        cluster_transforms: dict[_poss_cluster_args_literal, Any] | None = None,
    ):
        self.annotated_pages = annotations
        # temp
        self._orders = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self._transforms: OCROnTheFlyTransformPack | None = None
        self._update_orders(orders)  # the three previous attributes are updated here

        self._intraparagraph_transforms = (
            list() if intraparagraph_transforms is None else intraparagraph_transforms
        )
        self._interparagraph_transforms = (
            list() if interparagraph_transforms is None else interparagraph_transforms
        )

        self.cluster_params = _default_cluster_parameters.copy()
        self.cluster_params.update(
            cluster_transforms if cluster_transforms is not None else dict()
        )

        # TODO: implement layout generation: perhaps as a wrapper class of OCRDataset that has 2 different annotation lists:
        # one is the original and the other is the layout-generated one. The rest should be more or less the same.

    def __repr__(self):
        return f"<OCRDataset ({len(self)} samples: {len(self.annotated_pages)} pages using orders {self.orders}>"

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
            self._use_paragraphs = "paragraph" in new_orders
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
            selected_box_ids = list(ann.lines.keys())
            order = "full"
            identifyer = f"pg{page_idx}"
        else:
            par_prefix = self._par_prefix_sums[page_idx]
            par_idx = int(np.searchsorted(par_prefix, rel_idx, side="right"))
            par_offset = int(par_prefix[par_idx - 1]) if par_idx > 0 else 0
            item_idx = rel_idx - par_offset

            paragraph = ann.paragraphs[par_idx]
            line_ids = paragraph.line_ids
            num_lines = len(line_ids)

            curr = item_idx
            selected_box_ids: Optional[list[str]] = None

            for order in self._orders:
                n_windows: int = max(0, num_lines - order + 1)
                if curr < n_windows:
                    start_line = curr
                    selected_box_ids = line_ids[start_line : start_line + order]
                    identifyer = str(
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

        synthetic_img, synthetic_transcription, sindex = ann.synthetic_sample(
            selected_box_ids,
            tight_layout=self.cluster_params["tight_layout"],
            margin_size_px=self.cluster_params["margin_size_px"],
            on_the_fly_transform_pack=self._transforms,
        )

        # TODO: improve context generation - implement the use_previous_page_in_context cluster parameter here
        context = ann.synthetic_transcription("all")[:sindex]
        return {
            "image": synthetic_img,
            "text": synthetic_transcription,
            "sindex": sindex,
            "context": context,
            "order": order,
            "id": identifyer,
            "page_id": ann.task_id,
        }

    @staticmethod
    def from_split(
        *groups_of_annotations: list[AnnotatedPage],
        p: float,
        orders: orders_type,
        orders_to_split_with: orders_type | None = None,
    ):
        """
        Generates two OCRDatasets (paradigmatically train and test) from various groups of AnnotatedPages.
        This could be used to generate splits that are balanced in difficulty, passing groups of annotations
        that differ in difficulty, or in any other characteristic, to get splits homogeneous in that characteristic.
        The split is done taking the number of samples in each annotation considering only those samples that
        are of order orders_to_split_with (or 'orders' directly, if the former is not given a value.)

        ann_group_1 = [...]
        ann_group_2 = [...]
        ann_group_3 = [...]

        train, test = OCRDataset.from_split(ann_group_1, ann_group_2, ann_group_3, p = 0.95, orders_to_split_with = [1], orders = [1,2,3,4,5])

        produces a split that

        """
        train = []
        test = []

        for annotations in groups_of_annotations:
            train_i, test_i = OCRDataset.montecarlo_ann_split(
                annotations,
                p,
                orders=orders if orders_to_split_with is None else orders_to_split_with,
            )

            train += train_i
            test += test_i
        return OCRDataset(train, orders=orders), OCRDataset(test, orders=orders)

    def set_transform(self, transform: OCROnTheFlyTransformPack | None):
        self._transforms = transform

    @staticmethod
    def samples_in_annotation(
        ann: AnnotatedPage, orders: Collection[int | Literal["paragraph", "full"]]
    ):
        return (
            sum(
                sum(
                    max(0, len(paragraph) - order + 1)
                    for order in orders
                    if isinstance(order, int)
                )
                for paragraph in ann.paragraphs
            )
            + len(ann.paragraphs) * (("paragraph") in orders)
            + ("full" in orders)
        )

    @staticmethod
    def montecarlo_ann_split(
        annotations: list[AnnotatedPage],
        p=0.95,
        orders: Collection[int | Literal["paragraph", "full"]] = [1],
        n_trials: int = 1000,
    ) -> tuple[list[AnnotatedPage], list[AnnotatedPage]]:

        print(f"Performing Monte Carlo page split with {n_trials} trials")

        weights = [
            (i, OCRDataset.samples_in_annotation(ann, orders))
            for i, ann in enumerate(annotations)
        ]

        total_samples = sum(weight for _, weight in weights)

        if total_samples == 0:
            raise ValueError("Total number of samples is zero.")

        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        target_a = total_samples * p

        best_error = float("inf")
        best_pages: set[int] = set()

        for _ in range(n_trials):
            candidate_pages = {i for i, _ in weights if np.random.rand() < p}

            candidate_samples = sum(
                weight for i, weight in weights if i in candidate_pages
            )

            error = abs(candidate_samples - target_a)

            if error < best_error:
                best_error = error
                best_pages = candidate_pages

                # Exact match, so there is no reason to keep searching.
                if error == 0:
                    break

        train_annotations = [
            annotations[i] for i in range(len(annotations)) if i in best_pages
        ]

        test_annotations = [
            annotations[i] for i in range(len(annotations)) if i not in best_pages
        ]

        train_samples = sum(
            OCRDataset.samples_in_annotation(annotation, orders)
            for annotation in train_annotations
        )

        test_samples = sum(
            OCRDataset.samples_in_annotation(annotation, orders)
            for annotation in test_annotations
        )
        return train_annotations, test_annotations
