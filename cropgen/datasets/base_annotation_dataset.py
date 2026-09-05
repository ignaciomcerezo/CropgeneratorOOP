from cropgen.transforms.transforms import (
    LinewiseTransform,
    IntraparagraphTransform,
    InterparagraphTransform,
)
from warnings import warn
from cropgen.datasets.ocr_transform_pack import OCRTransformPack
from cropgen.ocr_units import OCRPage
from typing import Sequence, Optional, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Collection, Literal, get_args, Any
import numpy as np

orders_type = Collection[int | Literal["paragraph", "page"]]

_poss_cluster_args_literal = Literal[
    "tight_layout",
    "margin_size_px",
    "use_previous_page_in_context",
    "avoid_intersections",
    "overlay_polygons",
    "overlay_mbr",
]

_default_cluster_param_values = (
    True,
    {"left": 0, "right": 0, "top": 0, "bottom": 0},
    False,
    True,
    False,
    False,
)

_default_cluster_parameters: dict[_poss_cluster_args_literal, Any] = {
    arg: value
    for arg, value in zip(
        get_args(_poss_cluster_args_literal), _default_cluster_param_values
    )
}

T = TypeVar("T")


class BaseAnnotationDataset(Dataset, ABC):
    _annotated_pages: Sequence[OCRPage]
    _orders: list[int]
    _use_paragraphs: bool
    _use_full_pages: bool
    _transforms: OCRTransformPack
    _cluster_params = field(default_factory=lambda: _default_cluster_parameters.copy())

    @property
    def pages(self) -> list[str | None]:
        return [ann.page for ann in self._annotated_pages]

    @property
    def ids(self) -> list[int]:
        return [ann.task_id for ann in self._annotated_pages]

    @property
    def orders(self):
        return (
            self._orders
            + ["paragraph"] * self._use_paragraphs
            + ["page"] * self._use_full_pages
        )

    @orders.setter
    def orders(self, value: Sequence[int | Literal["paragraph", "page"]]):
        self._update_orders(value)

    @property
    def cluster_params(self):
        return tuple([(key, value) for key, value in self._cluster_params.items()])

    def set_cluster_param(
        self, cluster_param_name: _poss_cluster_args_literal, value: Any
    ):
        if cluster_param_name not in _default_cluster_parameters:
            raise ValueError(
                f"Unknown cluster parameter '{cluster_param_name}' (expected one of {_default_cluster_parameters})."
            )
        if value is not None:
            self._cluster_params[cluster_param_name] = value
        else:
            self._cluster_params[cluster_param_name] = _default_cluster_parameters[
                cluster_param_name
            ]
        if cluster_param_name == "avoid_intersections":
            self._transforms._avoid_intersections = value

    def _update_orders(
        self,
        new_orders: Collection[int | Literal["paragraph", "page"]],
    ):
        for order in new_orders:
            if (
                isinstance(order, float)
                or (
                    isinstance(order, str)
                    and ((order != "page") and (order != "paragraph"))
                )
                or (isinstance(order, int) and (order < 1))
            ):
                raise ValueError(
                    f"Value '{order}' found inside value for orders. "
                    "Only ints > 0, 'paragraph' and 'page' are acceptable orders."
                )

        pseudo_old_orders: set[int | Literal["paragraph", "page"]] = set(self._orders)

        if self._use_paragraphs:
            pseudo_old_orders.add("paragraph")
        if self._use_full_pages:
            pseudo_old_orders.add("page")

        if set(new_orders) != pseudo_old_orders:
            self._orders = [x for x in new_orders if isinstance(x, int)]
            self._use_paragraphs = "paragraph" in new_orders
            self._use_full_pages = "page" in new_orders
            self._recalculate_size_and_sampling_params()

    def update_stage(
        self,
        new_orders: list[int],
    ):
        self._update_orders(new_orders)

    @staticmethod
    def _is_single_paragraph_page(ann: OCRPage) -> bool:
        """
        Whether the page consists of exactly one paragraph.

        Such a page represents the same set of lines whether it is sampled
        as a paragraph or as a complete page.
        """
        return len(ann.paragraphs) == 1

    def _page_is_additional_sample(self, ann: OCRPage) -> bool:
        """
        Whether the complete-page sample should be counted separately.

        If a page has exactly one paragraph and both paragraph and page
        sampling are enabled, the paragraph and page refer to the exact
        same sample and must therefore only be counted once.
        """
        return self._use_full_pages and not (
            self._use_paragraphs and self._is_single_paragraph_page(ann)
        )

    def _recalculate_size_and_sampling_params(self):
        page_sample_counts: list[int] = []
        par_prefix_sums: list[np.ndarray] = []
        page_prefix_sums: list[int] = []
        running_total: int = 0

        for ann in self._annotated_pages:
            par_counts: list[int] = []

            for par in ann.paragraphs:
                par_len = len(par)

                # Number of valid sliding windows of each integer order,
                # plus one sample if paragraph sampling is enabled.
                ell = sum(max(0, par_len - order + 1) for order in self._orders) + int(
                    self._use_paragraphs
                )

                par_counts.append(ell)

            page_par_samples = sum(par_counts)

            # A single-paragraph page is already represented by its paragraph
            # sample when both paragraph and page sampling are enabled.
            page_is_additional_sample = self._page_is_additional_sample(ann)

            page_total_samples = page_par_samples + int(page_is_additional_sample)

            par_prefix_sums.append(np.cumsum(par_counts, dtype=np.int64))
            page_sample_counts.append(page_total_samples)

            running_total += page_total_samples
            page_prefix_sums.append(running_total)

        self._size = running_total
        self._page_sample_counts = page_sample_counts
        self._page_prefix_sums = np.array(
            page_prefix_sums,
            dtype=np.int64,
        )
        self._par_prefix_sums = par_prefix_sums

    def __len__(self) -> int:
        return self._size

    def _gets_ann_ids_order_and_identifier(self, index: int) -> tuple[
        OCRPage,
        Sequence[str],
        int | Literal["paragraph", "page"],
        str,
    ]:
        """
        Chooses an annotation instance and specific lines according to the
        available orders.

        Each sample is chosen uniformly.

        When a page contains exactly one paragraph and both paragraph and
        page sampling are enabled, that page contributes only one sample.
        The sample is represented as the paragraph sample because its line
        IDs are identical to those of the complete page.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        page_idx = int(
            np.searchsorted(
                self._page_prefix_sums,
                index,
                side="right",
            )
        )

        page_offset = int(self._page_prefix_sums[page_idx - 1]) if page_idx > 0 else 0

        rel_idx = index - page_offset

        ann: OCRPage = self._annotated_pages[page_idx]

        # A page sample only occupies its own index when it was actually
        # counted as an additional sample.
        page_is_additional_sample = self._page_is_additional_sample(ann)

        if (
            page_is_additional_sample
            and rel_idx == self._page_sample_counts[page_idx] - 1
        ):
            selected_line_ids = list(ann.lines.keys())
            order = "page"
            identifier = f"pg{page_idx}"

        else:
            par_prefix = self._par_prefix_sums[page_idx]

            par_idx = int(
                np.searchsorted(
                    par_prefix,
                    rel_idx,
                    side="right",
                )
            )

            par_offset = int(par_prefix[par_idx - 1]) if par_idx > 0 else 0

            item_idx = rel_idx - par_offset

            paragraph = ann.paragraphs[par_idx]
            line_ids = paragraph.line_ids
            num_lines = len(line_ids)

            curr = item_idx
            selected_line_ids: Optional[list[str]] = None

            for order in self._orders:
                n_windows: int = max(0, num_lines - order + 1)

                if curr < n_windows:
                    start_line = curr

                    selected_line_ids = line_ids[start_line : start_line + order]

                    identifier = (
                        f"pg{page_idx}par{par_idx}L{start_line}"
                        f"{start_line + order - 1}" * (order > 1)
                    )
                    break

                curr -= n_windows

            if selected_line_ids is None and self._use_paragraphs:
                selected_line_ids = line_ids
                order = "paragraph"
                identifier = f"pg{page_idx}par{par_idx}"

            if selected_line_ids is None:
                raise RuntimeError(f"Failed to resolve sample target for index {index}")

        return ann, selected_line_ids, order, identifier

    @property
    def transforms(self) -> OCRTransformPack | None:
        if self._transforms is None or self._transforms.is_identity:
            return None
        else:
            return self._transforms

    def add_transform(
        self,
        transform: (
            LinewiseTransform | IntraparagraphTransform | InterparagraphTransform | None
        ),
        probability: float = 1,
    ) -> None:
        if transform is not None:
            self._transforms.add_transform(transform, probability)

    def set_transform(
        self,
        *transform_probability_pairs: tuple[
            LinewiseTransform
            | IntraparagraphTransform
            | InterparagraphTransform
            | None,
            float,
        ],
    ) -> None:

        for transform in (transform for transform, _ in transform_probability_pairs):
            if transform is not None and not isinstance(
                transform,
                (LinewiseTransform, IntraparagraphTransform, InterparagraphTransform),
            ):
                raise ValueError(
                    f"Only accepts LinewiseTransform, IntraparagraphTransform or InterparagraphTransform, got {type(transform)}"
                )
        transform: (
            None | IntraparagraphTransform | LinewiseTransform | InterparagraphTransform
        )
        self._transforms = OCRTransformPack(
            avoid_intersections=self._cluster_params["avoid_intersections"]
        )
        for transform, probability in transform_probability_pairs:
            if probability != 0:
                self.add_transform(transform, probability)

    @staticmethod
    def samples_in_annotation(
        ann: OCRPage,
        orders: Collection[int | Literal["paragraph", "page"]],
    ):
        use_paragraphs = "paragraph" in orders
        use_full_pages = "page" in orders

        paragraph_samples = sum(
            max(0, len(paragraph) - order + 1)
            for paragraph in ann.paragraphs
            for order in orders
            if isinstance(order, int)
        )

        paragraph_samples += len(ann.paragraphs) if use_paragraphs else 0

        # If the page consists of exactly one paragraph and both sampling
        # modes are enabled, the page and paragraph are the same sample.
        page_samples = int(
            use_full_pages
            and not (
                use_paragraphs and BaseAnnotationDataset._is_single_paragraph_page(ann)
            )
        )

        return paragraph_samples + page_samples

    @staticmethod
    def montecarlo_ann_split(
        annotations: list[OCRPage],
        p=0.95,
        orders: Collection[int | Literal["paragraph", "page"]] = [1],
        n_trials: int = 1000,
    ) -> tuple[list[OCRPage], list[OCRPage]]:

        print(f"Performing Monte Carlo page split with {n_trials} trials")

        weights = [
            (
                i,
                BaseAnnotationDataset.samples_in_annotation(
                    ann,
                    orders,
                ),
            )
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

        return train_annotations, test_annotations

    @classmethod
    def from_split(
        cls,
        *groups_of_annotations: list[OCRPage],
        p: float,
        orders: orders_type,
        orders_to_split_with: orders_type | None = None,
    ) -> tuple[T, T]:
        """
        Generates two datasets (paradigmatically train and test) from
        various groups of AnnotatedPages.

        This could be used to generate splits that are balanced in
        difficulty, passing groups of annotations that differ in difficulty,
        or in any other characteristic, to get splits homogeneous in that
        characteristic.

        The split is done taking the number of samples in each annotation
        considering only samples that are of order orders_to_split_with
        (or 'orders', if the former is not given a value).

        Example:

            ann_group_1 = [...]
            ann_group_2 = [...]
            ann_group_3 = [...]

            train, test = OCRDataset.from_split(
                ann_group_1,
                ann_group_2,
                ann_group_3,
                p=0.95,
                orders_to_split_with=[1],
                orders=[1, 2, 3, 4, 5],
            )

        This produces a split that has approximately 0.95 of groups 1, 2,
        and 3 in train, with the remaining annotations in test.
        """

        train = []
        test = []

        for annotations in groups_of_annotations:
            train_i, test_i = cls.montecarlo_ann_split(
                annotations,
                p,
                orders=(
                    orders if orders_to_split_with is None else orders_to_split_with
                ),
            )

            train += train_i
            test += test_i

        # TODO: this is not too idiomatic: we are not explicitly telling
        # Python that heirs must have an __init__ of this type...
        return (
            cls(
                train,  # ty: ignore[too-many-positional-arguments]
                orders=orders,  # ty: ignore[unknown-argument]
            ),
            cls(
                test,  # ty: ignore[too-many-positional-arguments]
                orders=orders,  # ty: ignore[unknown-argument]
            ),
        )  # ty: ignore[invalid-return-type]

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
