from copy import deepcopy
from shapely.geometry import Point, Polygon
from cropgen.processing.line import Line
from cropgen.processing.paragraph import Paragraph
from typing import Literal, Optional, TYPE_CHECKING, Any, Sequence, Callable
import re
from collections.abc import Iterable
from shapely.affinity import translate
import numpy as np
from PIL import Image, ImageDraw

from cropgen.processing.helpers.helper_to_classes import (
    get_connected_components,
    get_union_rect,
    subdictionary,
)
from cropgen.processing.helpers.image_processing import (
    separate_background_and_stroke,
    crop_or_resize,
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
from cropgen.shared.default_parameters import (
    MAX_IMG_DIM,
    OPERATIONS_IMG_DIM,
    INPAINTING_IMG_DIM,
)
from cropgen.shared.display import display

ocr_transform = Callable[
    [list[Image.Image], list[Polygon]],
    tuple[list[Image.Image], list[Polygon]],
]


class AnnotatedPage:
    """
    Single annotated page, gathers all the information about lines and paragraphs, and implements some
    methods to create synthetic annotations by using only some of the lines in a page (.synthetic_sample).
    """

    n_annotation_errors: int = 0
    working_img_longest_side: int = MAX_IMG_DIM
    _stroke_separation_img_longest_side: int = OPERATIONS_IMG_DIM
    _inpainting_img_longest_side: int = INPAINTING_IMG_DIM
    __slots__ = (
        "lines",
        "background",
        "stroke",
        "task_id",
        "_graph",
        "last_update_time",
        "completer",
        "updater",
        "paragraphs",
        "annotation_unique_id",
        "line_separator",
        "page",
    )

    def __init__(
        self,
        ann: SimplifiedAnnotation,
        img: Image.Image,
        usernames_labelstudio: list[str] | None = None,
        line_separtor: str = "\n",
        page: str | None = None,
        stroke: Image.Image | None = None,
        background: Image.Image | None = None,
    ):

        assert (
            usernames_labelstudio is not None
        ), "A Label Studio username list must be provided."

        self.task_id = ann.task
        results: list[SimplifiedResultItem] = ann.result

        if img.mode != "L":
            img = img.convert("L")

        if (stroke is None) or (background is None):
            self.background, self.stroke = separate_background_and_stroke(
                img,
                out_longest_side=AnnotatedPage.working_img_longest_side,
                processing_longest_side=AnnotatedPage._stroke_separation_img_longest_side,
                inpaint_longest_side=AnnotatedPage._inpainting_img_longest_side,
            )
        else:
            self.stroke = stroke
            self.background = background

        self.page = page
        self.line_separator = line_separtor

        self._setup_lines(results)

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

    @staticmethod
    def from_paragraphs(
        paragraphs: list[Paragraph],
        task_id: int,
        background: Optional[Image.Image] = None,
        stroke: Optional[Image.Image] = None,
        line_separator: str = "\n",
        completer: str = "Unknown",
        updater: str = "Unknown",
        last_update_time: str = "",
        annotation_unique_id: Optional[int] = None,
        page: str | None = None,
    ) -> "AnnotatedPage":
        """
        Alternate constructor that builds an AnnotatedPage directly from a list of already
        assembled Paragraph instances, bypassing the init pipeline.
        """
        if not paragraphs:
            raise ValueError(
                "Cannot build an AnnotatedPage from an empty list of paragraphs."
            )

        ann: "AnnotatedPage" = object.__new__(AnnotatedPage)

        ann.paragraphs = sorted(
            deepcopy(paragraphs), key=lambda paragraph: paragraph.lines[0].top
        )
        for index, paragraph in enumerate(ann.paragraphs):
            paragraph.index = index
        ann.task_id = task_id
        ann.line_separator = line_separator
        ann.background = (
            background if background is not None else Image.fromarray(np.zeros((1, 1)))
        )
        ann.stroke = stroke if stroke is not None else Image.fromarray(np.zeros((1, 1)))
        ann.completer = completer
        ann.updater = updater
        ann.last_update_time = last_update_time
        ann.annotation_unique_id = (
            annotation_unique_id
            if annotation_unique_id is not None
            else hash(last_update_time)
        )
        ann.page = page

        ann.lines = {}
        for paragraph in ann.paragraphs:
            for line in paragraph.lines:
                if line.id in ann.lines:
                    raise ValueError(
                        f"Duplicate line id {line.id!r} found while combining paragraphs "
                        "into a single AnnotatedPage."
                    )
                ann.lines[line.id] = line

        graph: dict[str, set[str]] = {}
        for paragraph in ann.paragraphs:
            if paragraph.subgraph is None:
                raise ValueError(
                    "Every paragraph passed to from_paragraphs must carry a subgraph in "
                    "order to rebuild the page's intersection graph."
                )
            for line_id, neighbours in paragraph.subgraph.items():
                graph[line_id] = set(neighbours)
        ann._graph = graph

        ann._correct_text_and_set_sindices()

        return ann

    @staticmethod
    def combine_annotations(*annotations: "AnnotatedPage") -> "AnnotatedPage":
        """
        Combines several AnnotatedPage instances into a single new one.

        All of the paragraphs from every given annotation are gathered as-is: their lines,
        subgraphs, indices and starting indices are left completely untouched. The only thing
        this method does with the paragraphs themselves is reorder them, top to bottom, using
        the top coordinate of each paragraph's topmost line (i.e. `paragraph.top`, which is
        by construction the minimum `.top` among the paragraph's lines).

        The rest of the resulting AnnotatedPage's metadata (background, stroke, task_id...)
        is inherited from the first annotation passed in.
        """
        if not annotations:
            raise ValueError("combine_annotations needs at least one AnnotatedPage.")

        all_paragraphs = [
            paragraph
            for annotation in annotations
            for paragraph in annotation.paragraphs
        ]

        if (not len(set(ann.page for ann in annotations)) == 1) or (
            not len(set(ann.task_id for ann in annotations))
        ):
            raise ValueError(
                "Can only commbine annotations from the same page and task."
            )

        all_paragraphs.sort(key=lambda paragraph: paragraph.top)

        first = annotations[0]

        return AnnotatedPage.from_paragraphs(
            all_paragraphs,
            task_id=first.task_id,
            background=first.background,
            stroke=first.stroke,
            line_separator=first.line_separator,
            completer=first.completer,
            updater=first.updater,
            last_update_time=first.last_update_time,
            annotation_unique_id=first.annotation_unique_id,
            page=annotations[0].page,
        )

    @property
    def order(self) -> int:
        """Total number of lines"""
        return len(self.graph)

    @property
    def graph(self) -> dict[str, set[str]]:
        """Line's polygon annotation intersection graph which keys are the ImageBox(es) ids"""
        return self._graph

    def _setup_lines(self, results: list[SimplifiedResultItem]) -> None:
        """
        Generates all Line instances from the Label Studio results.
        """
        id2boxres: dict[str, RectangleResult | PolygonResult] = {
            r.id: r for r in results if isinstance(r, (RectangleResult, PolygonResult))
        }

        id2txtres: dict[str, SimplifiedTextCorrectionResult] = {
            r.id: r for r in results if isinstance(r, SimplifiedTextCorrectionResult)
        }

        self.lines: dict[str, Line] = dict()

        def is_fragment_with_error(identifyer: str) -> bool:
            if identifyer in id2txtres:
                return True
            elif identifyer in id2boxres:
                return False
            else:
                raise ValueError(
                    f"A relation in {self.task_id} connects a non-box non-fragment object."
                )

        seen_boxes: set[str] = set()
        seen_fragments: set[str] = set()
        for r in results:
            if isinstance(r, RelationResult):  # if the result is a relation
                source_id, target_id = r.from_id, r.to_id

                source_is_fragment = is_fragment_with_error(source_id)
                target_is_fragment = is_fragment_with_error(target_id)

                match (source_is_fragment, target_is_fragment):
                    case (False, True):
                        box_id, txt_id = source_id, target_id
                    case (True, False):
                        txt_id, box_id = source_id, target_id
                    case _:
                        # error: box to box OR fragment to fragment association
                        obj_type = ["box", "fragment"][source_is_fragment]
                        raise ValueError(
                            f"(Task {self.task_id}) {obj_type} to {obj_type} association:"
                            f"{obj_type} {source_id} -> {obj_type} {target_id}."
                        )

                if box_id in seen_boxes:
                    raise ValueError(
                        f"(Task {self.task_id}) box {box_id} has multiple associated fragments."
                    )
                if txt_id in seen_fragments:
                    raise ValueError(
                        f"(Task {self.task_id}) fragment {txt_id} has multiple associated boxes."
                    )

                seen_boxes.add(box_id)
                seen_fragments.add(txt_id)

                boxres = id2boxres[box_id]
                txtres = id2txtres[txt_id]

                line = Line.from_matching_ann_results(
                    boxres, txtres, self.task_id, self.stroke
                )

                self.lines[line.id] = line

        if (len(self.lines) != len(id2boxres)) or (len(self.lines) != len(id2txtres)):
            raise ValueError(
                f"(Task {self.task_id}) Some boxes/fragments have no associated other."
            )

    def _setup_graph_and_paragraphs(self):
        """
        Builds the intersection graph given by the polygons of the ImageBoxes.
        Also groups them into paragraphs (connected components) and sorts the
        boxes in their reading order.
        """
        lines = list(self.lines.values())

        adj: dict[str, set[str]] = {line.id: set() for line in lines}

        for i, line_a in enumerate(lines):
            for line_b in lines[i + 1 :]:
                if line_a.polygon.intersects(line_b.polygon):
                    adj[line_a.id].add(line_b.id)
                    adj[line_b.id].add(line_a.id)

        self._graph = adj

        connected_components = get_connected_components(adj)

        line_ccs = [
            [self.lines[line_id] for line_id in component]
            for component in connected_components
        ]

        subgraphs_ccs = [
            subdictionary(component, subdictionary(component, self.graph))
            for component in connected_components
        ]

        self.paragraphs: list[Paragraph] = sorted(
            [
                Paragraph(box_cc, task_id=self.task_id, subgraph=subgraph)
                for (box_cc, subgraph) in zip(line_ccs, subgraphs_ccs)
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

            if len(regularized_transcriptions) != len(paragraph.lines):
                raise ValueError(
                    "The number of lines after text regularization and the number of original lines do not match."
                )

            for line, new_transcription in zip(
                paragraph.lines, regularized_transcriptions
            ):
                line.text = regularize_line(new_transcription)
                line.starting_index = sindex
                sindex += len(line.text) + len(self.line_separator)

    def __repr__(self):
        pageif = (
            f"(page {self.page})" if self.page is not None else "(unknown page name)"
        )
        return (
            f"<Annotation of task {self.task_id} {pageif} of order {self.order}. Completed by {self.completer}, "
            f"last updated by {self.updater} at {self.last_update_time}>"
        )

    def synthetic_starting_index(
        self, line_ids: set[str] | list[str] | Literal["all"]
    ) -> int:
        if None in set(self.lines[line_id].starting_index for line_id in line_ids):
            raise ValueError(
                "Cannot compute transcription or sindex for an unordered group of lines."
            )
        starting_index: int = min(
            self.lines[line_id].starting_index
            for line_id in line_ids  # ty: ignore[invalid-argument-type]
        )

        return starting_index

    def synthetic_transcription(
        self,
        line_ids: set[str] | list[str] | Literal["all"],
    ) -> str:
        lines = (
            [self.lines[line_id] for line_id in line_ids]
            if line_ids != "all"
            else list(self.lines.values())
        )

        # using .starting_index has the same ordering as the reading order in image_boxes by design
        lines: list[Line] = sorted(
            lines, key=lambda x: x.starting_index
        )  # ty: ignore[no-matching-overload]

        return self.line_separator.join([line.text for line in lines])

    def synthetic_manuscript(
        self,
        line_ids: set[str] | list[str] | Literal["all"],
        *,
        tight_layout: bool = True,
        margin_size_px: int = 0,
        img_poly_transform: ocr_transform | None = None,
        overlay_polygons: bool = False,
        overlay_mbr: bool = False,
    ) -> tuple[Image.Image, list[Polygon]]:
        if line_ids == "all":
            line_ids = set(self.lines.keys())

        lines = sorted(
            [self.lines[box_id] for box_id in line_ids],
            key=lambda line: line.starting_index,
        )  # ty: ignore[no-matching-overload]

        if not isinstance(line_ids, (set, list)):
            raise ValueError(
                f"line_ids must be a set[str], list[str] or Literal['all'], but got {type(line_ids)}"
            )

        if len(lines) != len(set(line_ids)):
            raise ValueError("Duplicate line_ids passed to synthetic_manuscript.")

        strokes = [line.stroke_crop for line in lines]
        polygons = [line.polygon for line in lines]

        if img_poly_transform is not None:
            strokes, polygons = img_poly_transform(strokes, polygons)

        if not strokes:
            return self.background.copy(), polygons

        min_x, min_y, max_x, max_y = get_union_rect(polygons)
        bg_w, bg_h = self.background.width, self.background.height

        if tight_layout:
            x0 = int(min_x) - margin_size_px
            xf = int(max_x) + 1 + margin_size_px
            y0 = int(min_y) - margin_size_px
            yf = int(max_y) + 1 + margin_size_px
            can_crop = True
        else:
            x0 = min(0, int(min_x) - margin_size_px)
            xf = max(bg_w, int(max_x) + 1 + margin_size_px)
            y0 = min(0, int(min_y) - margin_size_px)
            yf = max(bg_h, int(max_y) + 1 + margin_size_px)
            can_crop = False

        bg_np = np.asarray(self.background)
        transformed_bg = crop_or_resize(
            bg_np, x0=x0, xf=xf, y0=y0, yf=yf, can_crop=can_crop
        )

        canvas_h, canvas_w = transformed_bg.shape[:2]
        overlay = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for stroke_img, polygon in zip(strokes, polygons):
            poly_x0, poly_y0, _, _ = polygon.bounds

            paste_x = int(poly_x0 - x0)
            paste_y = int(poly_y0 - y0)

            stroke_rgba = np.asarray(stroke_img.convert("RGBA"), dtype=np.float32)
            stroke_val = stroke_rgba[..., 0]
            alpha = stroke_rgba[..., 3] / 255.0
            masked_stroke = stroke_val * alpha

            sw, sh = stroke_img.width, stroke_img.height

            src_x0 = max(0, -paste_x)
            src_y0 = max(0, -paste_y)
            src_x1 = min(sw, canvas_w - paste_x)
            src_y1 = min(sh, canvas_h - paste_y)

            dst_x0 = max(0, paste_x)
            dst_y0 = max(0, paste_y)
            dst_x1 = min(canvas_w, paste_x + sw)
            dst_y1 = min(canvas_h, paste_y + sh)

            if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                overlay[dst_y0:dst_y1, dst_x0:dst_x1] += masked_stroke[
                    src_y0:src_y1, src_x0:src_x1
                ]

        bg_float = transformed_bg.astype(np.float32)
        if bg_float.ndim == 3 and bg_float.shape[2] in (3, 4):
            final_array = np.clip(bg_float - overlay[..., None], 0, 255).astype(
                np.uint8
            )
        else:
            final_array = np.clip(bg_float - overlay, 0, 255).astype(np.uint8)

        img = Image.fromarray(final_array)

        if overlay_mbr or overlay_polygons:
            img = self._overlay_polygons_mbr(
                lines=lines,
                manuscript=img,
                tight_layout=tight_layout,
                overlay_polygons=overlay_polygons,
                overlay_mbr=overlay_mbr,
            )

        return img, polygons

    @staticmethod
    def _overlay_polygons_mbr(
        *,
        lines: list[Line],
        manuscript: Image.Image,
        tight_layout: bool,
        overlay_polygons: bool,
        overlay_mbr: bool,
    ):
        origin_x = int(min(line.polygon.bounds[0] for line in lines)) * tight_layout
        origin_y = int(min(line.polygon.bounds[1] for line in lines)) * tight_layout

        # Usamos el mismo anclaje que synthetic_manuscript para convertir coordenadas globales a locales.
        manuscript_rgb = manuscript.convert("RGB")

        draw = ImageDraw.Draw(manuscript_rgb)

        for line in lines:
            if overlay_polygons:
                polygon_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in line.polygon.exterior.coords
                ]
                draw.line(polygon_points, fill=(255, 0, 0), width=3)

            if overlay_mbr:
                mbr_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in line.polygon.minimum_rotated_rectangle.exterior.coords
                ]
                draw.line(mbr_points, fill=(0, 255, 0), width=3)

        return manuscript_rgb

    def synthetic_sample(
        self,
        line_ids: list["str"] | Literal["all"],
        *,
        tight_layout: bool = True,
        margin_size_px: int = 0,
        img_poly_transform: ocr_transform | None = None,
    ) -> tuple[Image.Image, str, int]:
        """
        Given a list of ImageBox ids, returns:
        - their synthetic manuscript PIL.Image given by .synthetic_manuscript,
        - the transcription corresponding to this image,
        - the starting index of this text in the page transcription.
        """

        if not line_ids:
            raise ValueError("Cannot create a synthetic sample with no lines.")
        elif line_ids == "all":
            line_ids = list(self.lines.keys())

        manuscript = self.synthetic_manuscript(
            line_ids,
            tight_layout=tight_layout,
            margin_size_px=margin_size_px,
            img_poly_transform=img_poly_transform,
        )[0]

        transcription = self.synthetic_transcription(line_ids)
        starting_index = self.synthetic_starting_index(line_ids)

        return manuscript, transcription, starting_index
