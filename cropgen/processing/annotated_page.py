from typing import Literal
import re
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from cropgen.processing.image_box import ImageBox
from cropgen.processing.paragraph import Paragraph
from cropgen.processing.text_fragment import TextFragment
from cropgen.processing.helpers.PairingErrors import (
    NoAssociationError,
    MultipleAssociationError,
    RepeatedSameAssociationError,
    SameToSameAssociation,
)
from cropgen.processing.helpers.helper_to_classes import (
    get_connected_components,
    compose_collage,
    subdictionary,
)
from cropgen.processing.helpers.text_background_separator import (
    separate_background_and_stroke,
)
from cropgen.processing.helpers.text_regularization import (
    regularize_text,
    regularize_line,
)
from cropgen.shared.LSTypedDicts.results import (
    RectangleResult,
    PolygonResult,
    RelationResult,
    TextRegionResult,
)
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedAnnotation,
    SimplifiedResultItem,
    SimplifiedTextCorrectionResult,
)
from cropgen.shared.default_parameters import MAX_IMG_DIM
from cropgen.shared.display import display


class AnnotatedPage:
    """
    Single annotated page, gathers all the information about image-boxes and text fragments with their
    correspondances, builds the adjacency graph, structures the lines into paragraphs and implements
    methods to create synthetic annotations by using only some of the lines in a page (.synthetic_sample).
    """

    n_annotation_errors: int = 0
    warn_unrotate: bool = True
    warn_process_images: bool = True
    max_img_dimension: int = MAX_IMG_DIM
    __slots__ = (
        "image_boxes",
        "text_fragments",
        "background",
        "stroke",
        "task_id",
        "__graph",
        "last_update_time",
        "completer",
        "updater",
        "paragraphs",
        "annotation_unique_id",
        "process_images",
        "line_separator",
    )

    def __init__(
        self,
        ann: SimplifiedAnnotation,
        img: Image.Image,
        usernames_labelstudio: list[str] | None = None,
        line_separtor: str = "\n",
        process_images: bool = True,
    ):

        if not (process_images) and AnnotatedPage.warn_process_images:
            self._warn_process_images()

        assert (
            usernames_labelstudio is not None
        ), "A Label Studio username list must be provided."

        self.task_id = int(ann.task)
        results: list[SimplifiedResultItem] = ann.result

        if process_images:
            w, h = img.size

            scale_factor = AnnotatedPage.max_img_dimension / max(w, h)
            img = img.resize(
                (
                    int(np.ceil(w * scale_factor)),
                    int(np.ceil(h * scale_factor)),
                )
            )

            img = img.convert("L")

            self.background, self.stroke = separate_background_and_stroke(img)
        else:
            # blanks
            self.background = self.stroke = Image.Image()

        img_results_list: list[RectangleResult | PolygonResult] = [
            r for r in results if isinstance(r, (RectangleResult, PolygonResult))
        ]

        txt_results_list: list[SimplifiedTextCorrectionResult] = [
            r for r in results if isinstance(r, SimplifiedTextCorrectionResult)
        ]

        self.image_boxes: dict[str, ImageBox] = {  # ImageBox set
            img_result.id: ImageBox.from_image_result(
                img_result, self.task_id, self.stroke
            )
            for img_result in img_results_list
        }

        self.text_fragments: dict[str, TextFragment] = {
            txt_result.id: TextFragment(
                id=txt_result.id,
                text=" ".join(txt_result.value.text).strip(),
                task_id=self.task_id,
            )
            for txt_result in txt_results_list
        }

        self.process_images = process_images
        self.line_separator = line_separtor

        self._setup_mappings(results)

        self.assert_pairing()  # all images have an associated text fragment and vice versa.

        self._setup_graph_and_paragraphs()

        # only pages that lay inside of a paragraph have an sindex
        self._correct_text_and_set_sindices()

        self.last_update_time = " ".join(
            ann.updated_at.replace("Z", "").split("T")
        )  # task's last update

        completer_index = ann.completed_by
        updater_index = ann.updated_by
        self.completer = (
            usernames_labelstudio[completer_index]
            if completer_index < len(usernames_labelstudio)
            else "Unknown"
        )
        self.updater = (
            usernames_labelstudio[updater_index]
            if updater_index < len(usernames_labelstudio)
            else "Unknown"
        )
        self.annotation_unique_id = ann.id

    @property
    def order(self) -> int:
        """Total number of lines"""
        return len(self.graph)

    @property
    def graph(self) -> dict[str, set[str]]:
        """Line's polygon annotation intersection graph which keys are the ImageBox(es) ids"""
        return self.__graph

    def _setup_mappings(self, results: list[SimplifiedResultItem]) -> None:
        """
        Associates an ImageBox to each TextFragment (and vice versa) following the relations
        made by the annotators.
        """

        for r in results:
            if isinstance(r, RelationResult):  # if the result is a relation
                source_id, target_id = r.from_id, r.to_id

                if (source_id in self.image_boxes) and (
                    target_id in self.text_fragments
                ):
                    # ImgB -> TxtF
                    box_id, fragment_id = source_id, target_id
                elif (source_id in self.text_fragments) and (
                    target_id in self.image_boxes
                ):
                    # TxtF -> ImgB
                    box_id, fragment_id = target_id, source_id
                elif (source_id in self.image_boxes) and (
                    target_id in self.image_boxes
                ):
                    AnnotatedPage.n_annotation_errors += 1
                    # ImgB -> ImgB (annotation error)
                    print(f"(Task {self.task_id}) ImageBox to ImageBox association:")
                    print(
                        f"Box {self.image_boxes[source_id].id} -> Box {self.image_boxes[target_id].id}."
                    )
                    continue
                elif (source_id in self.text_fragments) and (
                    target_id in self.text_fragments
                ):
                    AnnotatedPage.n_annotation_errors += 1
                    # TxtF -> TxtF (annotation error)
                    print(
                        f"(Task {self.task_id}) TextFragment to TextFragment association:"
                    )
                    print(
                        f"Fragment {self.text_fragments[source_id].id} to {self.text_fragments[target_id].id}."
                    )
                    continue
                else:
                    AnnotatedPage.n_annotation_errors += 1
                    # other type of error
                    print(f"(Task {self.task_id}) Asociación rara.")
                    continue

                image_box = self.image_boxes[box_id]
                text_fragment = self.text_fragments[fragment_id]

                image_box.associate_fragment(text_fragment)
                text_fragment.associate_box(image_box)

    def _setup_graph_and_paragraphs(self):
        """
        Builds the intersection graph given by the polygons of the ImageBoxes.
        Also groups them into paragraphs (connected components) and sorts the
        boxes in their reading order.
        """
        self.__graph: dict[str, set[str]] = self._build_intersection_graph()
        connected_components = get_connected_components(self.__graph)

        box_ccs = [
            [self.image_boxes[box_id] for box_id in component]
            for component in connected_components
        ]

        subgraphs_ccs = [
            subdictionary(component, subdictionary(component, self.graph))
            for component in connected_components
        ]
        self.paragraphs: list[Paragraph] = sorted(
            [
                Paragraph(box_cc, task_id=self.task_id, subgraph=subgraph)
                for (box_cc, subgraph) in zip(box_ccs, subgraphs_ccs)
            ]
        )

    def _correct_text_and_set_sindices(self):
        sindex = 0
        for paragraph_index, paragraph in enumerate(self.paragraphs):
            paragraph.index = paragraph_index
            temporary_separator = "\n\x00\n"
            raw_separated_transcription = paragraph.transcription(temporary_separator)
            regularized_transcriptions = regularize_text(
                raw_separated_transcription
            ).split(temporary_separator)

            if len(regularized_transcriptions) != len(paragraph.text_fragments):
                raise ValueError(
                    "The number of lines after text regularization and the number of original lines do not match."
                )

            for fragment, new_transcription in zip(
                paragraph.text_fragments, regularized_transcriptions
            ):
                fragment.text = regularize_line(new_transcription)
                fragment.starting_index = sindex
                sindex += len(fragment.text) + len(self.line_separator)

    def assert_pairing(self):
        """Checks that all boxes have a fragment and vice versa."""
        for fragment in self.text_fragments.values():
            if any(
                [isinstance(obj, TextFragment) for obj in fragment.associated_boxes]
            ):
                raise SameToSameAssociation(fragment)

            if len(set(fragment.associated_boxes)) != len(fragment.associated_boxes):
                raise RepeatedSameAssociationError(fragment)
            elif len(fragment.associated_boxes) > 1:
                raise MultipleAssociationError(fragment)
            elif len(fragment.associated_boxes) == 0:
                raise NoAssociationError(fragment)

        for box in self.image_boxes.values():
            if any([isinstance(obj, ImageBox) for obj in box.associated_fragments]):
                raise SameToSameAssociation(box)

            if len(set(box.associated_fragments)) != len(box.associated_fragments):
                raise RepeatedSameAssociationError(box)
            elif len(box.associated_fragments) > 1:
                raise MultipleAssociationError(box)
            elif len(box.associated_fragments) == 0:
                raise NoAssociationError(box)

    def __repr__(self):
        return f"<Annotation of task {self.task_id} of order {self.order}. Completed by {self.completer}, last updated by {self.updater} at {self.last_update_time}>"

    def _build_intersection_graph(self):
        """
        Genera el grafo de intersecciones de una anotación.
        Devuelve un diccionario de adyacencia {box_id: set(id_adyacentes)}.
        """
        adj = {image_box_id: set() for image_box_id in self.image_boxes}
        for i, box1 in enumerate(self.image_boxes.values()):
            for j, box2 in enumerate(self.image_boxes.values()):
                if j <= i:
                    continue

                if box1.polygon.intersects(box2.polygon):
                    adj[box1.id].add(box2.id)
                    adj[box2.id].add(box1.id)

        return adj

    def synthetic_manuscript(
        self,
        box_id_sequence: set[str] | list[str] | Literal["all"],
        tight_layout: bool = True,
        margin_size_px: int = 0,
    ) -> Image.Image:
        """
        Generates the collage of handwritten strokes given by a sequence of ImageBox ids, placing each
        crop in its original place on the page, and using the background of the page, cropped or resized,
        to fit.
        """

        if box_id_sequence == "all":
            box_id_sequence = set(self.image_boxes.keys())

        if not self.process_images:
            return Image.Image()

        if not isinstance(box_id_sequence, set):
            if len(box_id_sequence) != len(set(box_id_sequence)):
                raise ValueError("There are duplicate box ids in synthetic_manuscript.")
            box_id_sequence = set(box_id_sequence)

        subgraph_image_boxes = [self.image_boxes[box_id] for box_id in box_id_sequence]

        return compose_collage(
            subgraph_image_boxes,
            background=self.background,
            tight_layout=tight_layout,
            margin_size_px=margin_size_px,
        )

    def synthetic_sample(
        self,
        box_ids: list["str"] | Literal["all"],
        tight_layout: bool = True,
        margin_size_px: int = 0,
    ) -> tuple[Image.Image, str, int]:
        """
        Given a list of ImageBox ids, returns:
        - their synthetic manuscript PIL.Image given by .synthetic_manuscript,
        - the transcription corresponding to this image,
        - the starting index of this text in the page transcription.
        """

        collage = self.synthetic_manuscript(
            box_ids, tight_layout=tight_layout, margin_size_px=margin_size_px
        )
        if box_ids == "all":
            box_ids = list(self.image_boxes.keys())
        fragments = [self.image_boxes[box_id].fragment for box_id in box_ids]

        # using .starting_index has the same ordering as the reading order in image_boxes by design
        fragments: list[TextFragment] = sorted(
            fragments, key=lambda x: x.starting_index
        )  # ty:ignore[no-matching-overload]

        transcription = self.line_separator.join(
            [fragment.text for fragment in fragments]
        )
        if not fragments:
            raise ValueError(
                f"Cannot use synthetic sample with no associated fragments, Task ({self.task_id}) -> {box_ids=}"
            )

        if fragments[0].starting_index is None:
            raise ValueError(
                "The fragments provided have no starting index - something has failed internally."
            )

        sindex = int(fragments[0].starting_index)

        return collage, transcription, sindex

    def synthetic_sample_with_polygons(
        self,
        image_box_ids: list[str],
        represent_polygon: bool = True,
        represent_mbr: bool = True,
        polygon_color: tuple[int, int, int] = (255, 0, 0),
        mbr_color: tuple[int, int, int] = (0, 255, 0),
        line_width: int = 3,
        crop_to_fit: bool = False,
    ) -> Image.Image:
        """
        Generates a synthetic sample with the polygons and minimum area rotated bounding box
        represented.
        """
        if not self.process_images:
            return Image.Image()

        selected_ids = list(image_box_ids)
        if not selected_ids:
            raise ValueError(
                "No se puede representar una secuencia vacía de image_box_ids."
            )

        selected_boxes = [self.image_boxes[box_id] for box_id in selected_ids]

        collage = self.synthetic_manuscript(list(self.image_boxes.keys()), crop_to_fit)

        origin_x = (
            int(min(box.polygon.bounds[0] for box in self.image_boxes.values()))
            * crop_to_fit
        )
        origin_y = (
            int(min(box.polygon.bounds[1] for box in self.image_boxes.values()))
            * crop_to_fit
        )

        # Usamos el mismo anclaje que compose_collage para convertir coordenadas globales a locales.
        collage = collage.convert("RGB")

        draw = ImageDraw.Draw(collage)

        for box in selected_boxes:
            if represent_polygon:
                polygon_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in box.polygon.exterior.coords
                ]
                draw.line(polygon_points, fill=polygon_color, width=line_width)

            if represent_mbr:
                mbr_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in box.polygon.minimum_rotated_rectangle.exterior.coords
                ]
                draw.line(mbr_points, fill=mbr_color, width=line_width)

        return collage

    @classmethod
    def from_paragraphs(
        cls,
        paragraphs: list[Paragraph],
        task_id: int,
        background: Image.Image,
        last_update_time: str,
        completer: str,
        updater: str,
        annotation_unique_id: int,
        line_separator: str,
        process_images: bool,
    ) -> "AnnotatedPage":
        """
        Constructs a new AnnotatedPage instance from a list of paragraphs.
        """
        instance: AnnotatedPage = cls.__new__(cls)

        instance.task_id = task_id
        instance.last_update_time = last_update_time
        instance.background = background
        instance.completer = completer
        instance.updater = updater
        instance.annotation_unique_id = annotation_unique_id
        instance.line_separator = line_separator
        instance.process_images = process_images

        instance.paragraphs = paragraphs
        instance.image_boxes = {}
        instance.text_fragments = {}

        sindex = 0
        for paragraph_index, paragraph in enumerate(instance.paragraphs):
            paragraph.index = paragraph_index
            for box in paragraph.image_boxes:
                instance.image_boxes[box.id] = box
            for fragment in paragraph.text_fragments:
                fragment.starting_index = sindex
                sindex += len(fragment.text) + 1
                instance.text_fragments[fragment.id] = fragment

        instance.__graph = instance._build_intersection_graph()

        return instance

    @staticmethod
    def combine_annotations(*annotations: "AnnotatedPage") -> "AnnotatedPage":
        """
        Combines two annotations in a single one sorting their paragraphs. As an ordering,
        uses the vertical coordinate of the centroid of the first line of each paragraph.
        """

        if len(annotations) <= 1:
            raise ValueError("At least two annotations must be passed as arguments.")

        if not (len(set(ann.task_id for ann in annotations)) == 1):
            raise ValueError(
                "Cannot combine annotations from two different tasks: "
                f" {set(ann.task_id for ann in annotations)}"
            )

        if not (len(set(ann.line_separator for ann in annotations)) == 1):
            raise ValueError(
                "Cannot combine two annotations with different line separators: "
                f" {set(ann.line_separator for ann in annotations)}"
            )

        combined_paragraphs = sum((ann.paragraphs for ann in annotations), start=[])

        def _topmost_order(paragraph: Paragraph) -> tuple[float, float]:
            topmost_box = paragraph.image_boxes[0]
            c = topmost_box.centroid()
            return (c[1], c[0])

        combined_paragraphs.sort(key=_topmost_order)
        last_update_time = max(ann.last_update_time for ann in annotations)

        completer = "+".join(set(ann.completer for ann in annotations))
        updater = "+".join(set(ann.updater for ann in annotations))
        annotation_id = int("000".join(str(ann.task_id) for ann in annotations))
        process_images = all(ann.process_images for ann in annotations)
        line_separator = annotations[0].line_separator
        background = annotations[0].background

        combined_ann = AnnotatedPage.from_paragraphs(
            paragraphs=combined_paragraphs,
            task_id=annotations[0].task_id,
            background=background,
            last_update_time=last_update_time,
            completer=completer,
            updater=updater,
            annotation_unique_id=annotation_id,
            process_images=process_images,
            line_separator=line_separator,
        )
        combined_ann._correct_text_and_set_sindices()
        return combined_ann

    def _warn_unrotate(self):
        print(
            "[!!!] WARN: Using unrotate = True destroys the information about the crop's original "
            "position on the page and distorts the images due to the digital rotation.\n"
            "It also invalidates how paragraphs are formed, the global transcription, synthetic sample "
            "generation and starting indices.\n"
            "It is only to be used in case of manual revision of the immages and NOT for production."
        )
        AnnotatedPage.warn_unrotate = False

    def _warn_process_images(self):
        print(
            "[!!!] WARN: Image processing deactivated: all crops and sample's images will be empty.\n"
            "Only to be used to speed up testing that does not require images and NOT for production."
        )
        AnnotatedPage.warn_process_images = False
