from warnings import warn
from cropgen.transforms.transforms import LinewiseTransform
from cropgen.transforms.on_the_fly_transform_manager import OCROnTheFlyTransformPack
from dataclasses import dataclass
from multiprocessing import Value
from cropgen.transforms import (
    IntraparagraphTransform,
    InterparagraphTransform,
)
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
from torch.utils.data import Dataset
import numpy as np

orders_type = Collection[int | Literal["paragraph", "page"]]


_poss_cluster_args_literal = Literal[
    "tight_layout",
    "margin_size_px",
    "use_previous_page_in_context",
    "avoid_intersections",
]
_default_cluster_param_values = (True, 0, False, True)

_default_cluster_parameters: dict[_poss_cluster_args_literal, Any] = {
    arg: value
    for arg, value in zip(
        get_args(_poss_cluster_args_literal), _default_cluster_param_values
    )
}

_default_getitem_output_literal = Literal[
    "image",
    "text",
    "sindex",
    "context",
    "order",
    "id",
    "page_id",
]


class OCRDataset(Dataset):
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
        self._orders = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self._transforms: OCROnTheFlyTransformPack | None = None
        self._update_orders(orders)  # the three previous attributes are updated here

        self._cluster_params = _default_cluster_parameters.copy()
        self._cluster_params.update(
            cluster_transform_params if cluster_transform_params is not None else dict()
        )

    def __repr__(self):
        return f"<OCRDataset ({len(self)} samples: {len(self._annotated_pages)} pages using orders {self.orders}>"

    @property
    def pages(self) -> list[str | None]:
        return [ann.page for ann in self._annotated_pages]

    @property
    def ids(self) -> list[int]:
        return [ann.task_id for ann in self._annotated_pages]

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, value: Sequence[int | Literal["paragraph", "page"]]):
        self._update_orders(value)

    @property
    def cluster_params(self):
        return self._cluster_params.copy()

    def set_cluster_param(
        self, cluster_param_name: _poss_cluster_args_literal, value: Any
    ):
        if value is not None:
            self._cluster_params[cluster_param_name] = value
        else:
            self._cluster_params[cluster_param_name] = _default_cluster_parameters[
                cluster_param_name
            ]

    def _update_orders(
        self,
        new_orders: Collection[int | Literal["paragraph", "page"]],
    ):
        for order in new_orders:
            if (
                (isinstance(order, float))
                or (
                    isinstance(order, str)
                    and ((order != "page") and (order != "paragraph"))
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
            pseudo_old_orders.add("page")

        if set(new_orders) != pseudo_old_orders:
            self._orders = [x for x in new_orders if isinstance(x, int)]
            self._use_paragraphs = "paragraph" in new_orders
            self._use_full_pages = "page" in new_orders
            self._recalculate_size_and_probabilities()

    def update_stage(
        self,
        new_orders: list[int],
    ):
        self._update_orders(new_orders)

    def _recalculate_size_and_probabilities(self):
        page_sample_counts: list[int] = []
        par_prefix_sums: list[np.ndarray] = []
        page_prefix_sums: list[int] = []
        running_total: int = 0

        for ann in self._annotated_pages:
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

    def __getitem__(self, index: int):
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

        ann: AnnotatedPage = self._annotated_pages[page_idx]

        if self._use_full_pages and rel_idx == self._page_sample_counts[page_idx] - 1:
            selected_box_ids = list(ann.lines.keys())
            order = "page"
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
            tight_layout=self._cluster_params["tight_layout"],
            margin_size_px=self._cluster_params["margin_size_px"],
            img_poly_transform=self._transforms,
        )

        # TODO: improve context generation - implement the use_previous_page_in_context cluster parameter here
        if sindex > 0:
            context = ann.synthetic_transcription("all")[:sindex]
        else:
            context = ""

        sample: dict[_default_getitem_output_literal, Any] = {
            "image": synthetic_img,
            "text": synthetic_transcription,
            "sindex": sindex,
            "context": context,
            "order": order,
            "id": identifyer,
            "page_id": ann.task_id,
        }

        if self._formatter is None:
            return sample
        else:
            return self._formatter(sample)

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

    def set_transform(
        self,
        transforms: OCROnTheFlyTransformPack | None,
    ):
        if transforms is None:
            self._transforms = transforms
            return
        elif isinstance(transforms, OCROnTheFlyTransformPack):
            if (
                transforms._avoid_intersections
                != self.cluster_params["avoid_intersections"]
            ):
                warn(
                    "Overwriting the transforms avoid_intersection parameter in acordance with cluster_params."
                )
                transforms._avoid_intersections = self.cluster_params[
                    "avoid_intersections"
                ]
            self._transforms = transforms
        else:
            raise ValueError(
                "Only accepts transforms as None (no transform) or instances of OCROnTheFlyTransformPack."
            )

    def set_formatter(
        self,
        formatter: Callable[[dict[_default_getitem_output_literal, Any]], Any] | None,
    ):
        self._formatter = formatter

    @staticmethod
    def samples_in_annotation(
        ann: AnnotatedPage, orders: Collection[int | Literal["paragraph", "page"]]
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
            + ("page" in orders)
        )

    @staticmethod
    def montecarlo_ann_split(
        annotations: list[AnnotatedPage],
        p=0.95,
        orders: Collection[int | Literal["paragraph", "page"]] = [1],
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
