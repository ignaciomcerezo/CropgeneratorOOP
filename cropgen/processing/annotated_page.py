from pathlib import Path
from cropgen.shared.PathBundle import PathBundle
from cropgen.shared.image_processing import crop_or_resize
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
import json

from cropgen.processing.helpers.helper_to_classes import (
    get_connected_components,
    get_union_rect,
    subdictionary,
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
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from cropgen.shared.display import display
from tqdm.auto import tqdm

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

    __slots__ = (
        "lines",
        "background",
        "task_id",
        "_graph",
        "completer",
        "updater",
        "paragraphs",
        "line_separator",
        "page",
    )

    def __init__(
        self,
        *,
        transcriptions: list[str],
        polygon_coords: list[list[tuple[float, float]]],
        line_ids: list[str],
        rotations: list[float],
        task_id: int,
        page: str,
        stroke: Image.Image,
        background: Image.Image,
        line_separtor: str = "\n",
        completer: str | None = None,
        updater: str | None = None,
        polygons_are_in_percentage: bool = True,
    ):

        self.background = background
        self.task_id = task_id
        self.page = page
        self.line_separator = line_separtor
        self.completer = completer if completer is not None else "Unknown"
        self.updater = updater if updater is not None else "Unknown"

        list_lines = self._setup_lines(
            polygon_coords,
            transcriptions,
            line_ids,
            rotations,
            stroke,
            polygons_are_in_percentage,
        )

        self.lines = {line.id: line for line in list_lines}

        self._setup_graph_and_paragraphs()

        # only pages that lay inside of a paragraph have an sindex
        self._correct_text_and_set_sindices()

    @property
    def image_dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the background in the format (width, height)."""
        return self.background.size

    @staticmethod
    def from_paragraphs(
        paragraphs: list[Paragraph],
        task_id: int,
        background: Image.Image,
        page: str,
        line_separator: str = "\n",
        completer: str = "Unknown",
        updater: str = "Unknown",
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
        ann.completer = completer
        ann.updater = updater
        ann.background = background
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
        background = annotations[0].background
        all_paragraphs.sort(key=lambda paragraph: paragraph.top)

        first = annotations[0]

        return AnnotatedPage.from_paragraphs(
            all_paragraphs,
            task_id=first.task_id,
            line_separator=first.line_separator,
            completer=first.completer,
            background=background,
            updater=first.updater,
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

    def _setup_lines(
        self,
        polygon_coords: list[list[tuple[float, float]]],
        transcriptions: list[str],
        line_ids: list[str],
        rotations: list[float],
        stroke: Image.Image,
        polygons_are_in_percentage: bool,
    ) -> list[Line]:

        if not (
            len(
                {
                    len(polygon_coords),
                    len(transcriptions),
                    len(line_ids),
                    len(rotations),
                }
            )
            == 1
        ):
            raise ValueError(
                "Inhomogeneous lengths of polygon_coords, transcriptions, line_ids and rotations."
            )

        if polygons_are_in_percentage:
            page_width, page_height = stroke.size
            polygons = []
            for i in range(len(polygon_coords)):
                polygon_coord = polygon_coords[i]

                polygons.append(
                    Polygon(
                        [
                            (p[0] * page_width / 100.0, p[1] * page_height / 100.0)
                            for p in polygon_coord
                        ]
                    )
                )
        else:
            polygons = [Polygon(polygon_coord) for polygon_coord in polygon_coords]

        lines = []

        for polygon, transcription, line_id, rotation in zip(
            polygons, transcriptions, line_ids, rotations
        ):
            lines.append(
                Line(
                    id=line_id,
                    stroke_crop=stroke,
                    polygon=polygon,
                    rotation=rotation,
                    task_id=self.task_id,
                    text=transcription,
                )
            )

        return lines

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
            f"last updated by {self.updater}.>"
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
        refit_polygons: bool = True,
        overlay_polygons: bool = False,
        overlay_mbr: bool = False,
    ) -> tuple[Image.Image, list[Polygon]]:

        assert margin_size_px >= 0, "The margin size cannot be negative."

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
        bg_w, bg_h = self.image_dimensions

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

        if refit_polygons:
            # displace the polygons to the new dimensions of the image
            polygons = [translate(polygon, -x0, -y0) for polygon in polygons]

        if overlay_mbr or overlay_polygons:
            img = self._overlay_polygons_mbr(
                refitted_polygons=(
                    polygons
                    if refit_polygons
                    else [translate(polygon, -x0, -y0) for polygon in polygons]
                ),
                manuscript=img,
                overlay_polygons=overlay_polygons,
                overlay_mbr=overlay_mbr,
            )

        return img, polygons

    @staticmethod
    def _overlay_polygons_mbr(
        *,
        refitted_polygons: list[Polygon],
        manuscript: Image.Image,
        overlay_polygons: bool,
        overlay_mbr: bool,
    ):

        manuscript_rgb = manuscript.convert("RGB")

        draw = ImageDraw.Draw(manuscript_rgb)

        for polygon in refitted_polygons:
            if overlay_polygons:
                # pointwise tranlation of all points
                polygon_points = [(x, y) for x, y in polygon.exterior.coords]
                draw.line(polygon_points, fill=(255, 0, 0), width=3)

            if overlay_mbr:
                mbr_points = [
                    (x, y) for x, y in polygon.minimum_rotated_rectangle.exterior.coords
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

    @staticmethod
    def from_paths(paths: PathBundle) -> list["AnnotatedPage"]:

        anns = []

        for metadata_filepath in tqdm(
            Path(paths.metadata_path).iterdir(), desc="Loading A.P. data from disk..."
        ):
            metadata_content: dict = json.loads(metadata_filepath.read_text())
            page: str = metadata_content["page"]
            task_id: int = metadata_content["task_id"]
            completer: str = metadata_content["completer"]
            updater: str = metadata_content["updater"]
            # subindex: int = metadata_content["subindex"]
            # ann_id  = metadata_content["ann_id"]
            # order = metadata_content["order"]
            ids: list[str] = metadata_content["ids"]
            polygons_are_in_percentage: bool = metadata_content[
                "polygons_are_in_percentage"
            ]

            transcriptions = json.loads(Path(metadata_content["txt_path"]).read_text())
            polygon_coords = json.loads(
                Path(metadata_content["polygons_path"]).read_text()
            )
            rotations = json.loads(Path(metadata_content["rotations_path"]).read_text())
            image_path = Path(metadata_content["image_path"])

            # stroke and background separation is not certain at this point
            stroke = Image.open(
                paths.stroke_images_path / (image_path.stem + image_path.suffix)
            )
            background = Image.open(
                paths.background_images_path / (image_path.stem + image_path.suffix)
            )

            anns.append(
                AnnotatedPage(
                    transcriptions=transcriptions,
                    polygon_coords=polygon_coords,
                    line_ids=ids,
                    rotations=rotations,
                    task_id=task_id,
                    page=page,
                    stroke=stroke,
                    background=background,
                    completer=completer,
                    updater=updater,
                    polygons_are_in_percentage=polygons_are_in_percentage,
                )
            )
        # TODO: PENDING FUSION OF SAME-PAGE ANNOTATIONS
        return anns
