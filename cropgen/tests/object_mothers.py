import random
from typing import Optional, Literal, Any

import numpy as np
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon

from cropgen.processing import AnnotatedPage, ImageBox, Paragraph, TextFragment
from cropgen.shared.LSTypedDicts.aggregates import (
    RawAnnotation,
    TaskData,
    LabelStudioTask,
)
from cropgen.shared.LSTypedDicts import (
    BaseResult,
    ImageBaseResult,
    TextRegionResult,
    TextCorrectionResult,
    RectangleResult,
    PolygonResult,
    RelationResult,
    SimplifiedTextCorrectionValue,
    SimplifiedTextCorrectionResult,
    SimplifiedAnnotation,
    SimplifiedTask,
    SimplifiedResultItem,
    TextRegionValue,
    TextCorrectionValue,
    RectangleValue,
    PolygonValue,
)

_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_text_chars = " â a á àbcdêeéèfghiîìíïjklmnoôóòpqrstuûúùvwxyzABCDEFGHIJKLMNOPQRSTUV.,:;"
_chars = _letters + "0123456789"
RelationDirection = Literal["right", "left", "bi"]


def _choose_randomly(string: str, length: int) -> str:
    return "".join(random.choice(string) for _ in range(length))


def random_letters(length: int = 8) -> str:
    return _choose_randomly(_letters, length)


def random_text(length: Optional[int] = None) -> str:
    length: int = length if length is not None else 40
    return _choose_randomly(_text_chars, length)


def random_chars(length: int = 8) -> str:
    return _choose_randomly(_chars, length)


def random_id(block_length: int = 4, n_blocks: int = 3) -> str:
    return "-".join([_choose_randomly(_chars, block_length) for _ in range(n_blocks)])


def mother_text_region_value(text: Optional[str] = None) -> TextRegionValue:
    start = random.randint(0, 10)
    end = start + random.randint(1, 10)
    text: str = text if text is not None else random_text(end - start)
    labels = [f"label{random.randint(1, 10)}"]
    return TextRegionValue(start=start, end=end, text=text, labels=labels)


def mother_text_correction_value(
    text: Optional[str] = None,
    start: Optional[int] = None,
    length: Optional[int] = None,
) -> TextCorrectionValue:
    if length and text:
        assert len(text) == length

    start: int = random.randint(0, 10) if start is None else start
    end = start + random.randint(1, 10) if length is None else length
    text_list = [text if text else random_text(end - start)]
    return TextCorrectionValue(start=start, end=end, text=text_list)


def mother_rectangle_value(
    x: Optional[float] = None,
    y: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    rotation: Optional[float] = None,
):
    x: float = x if x is not None else random.uniform(0, 100)
    y: float = y if y is not None else random.uniform(0, 100)
    width: float = width if width is not None else random.uniform(1, 100)
    height: float = height if height is not None else random.uniform(1, 100)
    rotation: float = rotation if rotation is not None else random.uniform(0, 360)
    rectanglelabels = [f"rect{random.randint(1, 10)}"]
    return RectangleValue(
        x=x,
        y=y,
        width=width,
        height=height,
        rotation=rotation,
        rectanglelabels=rectanglelabels,
    )


def mother_polygon_value(
    points: Optional[list[list[float | int]]] = None,
    n_vertices: Optional[int] = None,
    closed: Optional[bool] = None,
    polygonlabels: Optional[list[str]] = None,
) -> PolygonValue:
    n_points = n_vertices if n_vertices is not None else random.randint(3, 8)
    generated_points = [
        [random.uniform(0, 100), random.uniform(0, 100)]
        for _ in range(max(3, n_points))
    ]
    final_points: list[list[float | int]] = (
        [list(point) for point in points] if points is not None else generated_points
    )
    is_closed: bool = True if closed is None else closed
    if is_closed and final_points and final_points[0] != final_points[-1]:
        final_points = [*final_points, list(final_points[0])]
    final_polygonlabels: list[str] = (
        polygonlabels if polygonlabels is not None else [f"poly{random.randint(1, 10)}"]
    )
    return PolygonValue(
        points=final_points,
        closed=is_closed,
        polygonlabels=final_polygonlabels,
    )


def mother_base_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    type: Optional[str] = None,
    origin: Optional[str] = None,
) -> BaseResult:
    return BaseResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type=type if type is not None else f"type_{random.randint(1, 5)}",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
    )


def mother_image_base_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    type: Optional[str] = None,
    origin: Optional[str] = None,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    image_rotation: Optional[float | int] = None,
) -> ImageBaseResult:
    return ImageBaseResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type=type if type is not None else f"type_{random.randint(1, 5)}",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        original_width=(
            original_width if original_width is not None else random.randint(50, 500)
        ),
        original_height=(
            original_height if original_height is not None else random.randint(50, 500)
        ),
        image_rotation=(
            image_rotation if image_rotation is not None else random.uniform(0, 360)
        ),
    )


def mother_text_region_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    origin: Optional[str] = None,
    value: Optional[TextRegionValue] = None,
    value_text: Optional[str] = None,
) -> TextRegionResult:
    return TextRegionResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type="labels",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        from_name="txt_spans",
        value=value if value is not None else mother_text_region_value(text=value_text),
    )


def mother_text_correction_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    origin: Optional[str] = None,
    value: Optional[TextCorrectionValue] = None,
    value_text: Optional[str] = None,
    value_start: Optional[int] = None,
    value_length: Optional[int] = None,
) -> TextCorrectionResult:
    return TextCorrectionResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type="textarea",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        from_name="correction",
        value=(
            value
            if value is not None
            else mother_text_correction_value(
                text=value_text,
                start=value_start,
                length=value_length,
            )
        ),
    )


def mother_rectangle_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    origin: Optional[str] = None,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    image_rotation: Optional[float | int] = None,
    value: Optional[RectangleValue] = None,
    value_x: Optional[float] = None,
    value_y: Optional[float] = None,
    value_width: Optional[float] = None,
    value_height: Optional[float] = None,
    value_rotation: Optional[float] = None,
) -> RectangleResult:
    return RectangleResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type="rectanglelabels",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        original_width=(
            original_width if original_width is not None else random.randint(50, 500)
        ),
        original_height=(
            original_height if original_height is not None else random.randint(50, 500)
        ),
        image_rotation=(
            image_rotation if image_rotation is not None else random.uniform(0, 360)
        ),
        from_name="img_regions",
        value=(
            value
            if value is not None
            else mother_rectangle_value(
                x=value_x,
                y=value_y,
                width=value_width,
                height=value_height,
                rotation=value_rotation,
            )
        ),
    )


def mother_polygon_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    origin: Optional[str] = None,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
    image_rotation: Optional[float | int] = None,
    value: Optional[PolygonValue] = None,
    value_points: Optional[list[list[float | int]]] = None,
    value_n_vertices: Optional[int] = None,
    value_closed: Optional[bool] = None,
    value_polygonlabels: Optional[list[str]] = None,
) -> PolygonResult:
    return PolygonResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type="polygonlabels",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        original_width=(
            original_width if original_width is not None else random.randint(50, 500)
        ),
        original_height=(
            original_height if original_height is not None else random.randint(50, 500)
        ),
        image_rotation=(
            image_rotation if image_rotation is not None else random.uniform(0, 360)
        ),
        from_name="img_polygons",
        value=(
            value
            if value is not None
            else mother_polygon_value(
                points=value_points,
                n_vertices=value_n_vertices,
                closed=value_closed,
                polygonlabels=value_polygonlabels,
            )
        ),
    )


def mother_relation_result(
    from_id: Optional[str] = None,
    to_id: Optional[str] = None,
    direction: Optional[RelationDirection] = None,
) -> RelationResult:
    return RelationResult(
        from_id=from_id if from_id is not None else random_id(),
        to_id=to_id if to_id is not None else random_id(),
        type="relation",
        direction=(
            direction
            if direction is not None
            else random.choice(["right", "left", "bi"])
        ),
    )


def mother_raw_annotation(
    id: Optional[int] = None,
    completed_by: Optional[int] = None,
    result: Optional[list[Any]] = None,
    result_count: Optional[int] = None,
    was_cancelled: Optional[bool] = None,
    ground_truth: Optional[bool] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    lead_time: Optional[float] = None,
    unique_id: Optional[str] = None,
    bulk_created: Optional[bool] = None,
    task: Optional[int] = None,
    project: Optional[int] = None,
    updated_by: Optional[int] = None,
    text_region_result: Optional[TextRegionResult] = None,
    text_region_result_kwargs: Optional[dict[str, Any]] = None,
) -> RawAnnotation:

    created_at: str = created_at if created_at is not None else "2026-01-01"
    updated_at: str = updated_at if updated_at is not None else "2026-01-01"
    if result is None:
        result: list[TextRegionResult] = [
            (
                text_region_result
                if text_region_result is not None
                else mother_text_region_result(**(text_region_result_kwargs or {}))
            )
        ]
    resolved_result_count: int = (
        result_count if result_count is not None else int(random.randint(1, 5))
    )
    return RawAnnotation(
        id=id if id is not None else random.randint(1, 10000),
        completed_by=(
            completed_by if completed_by is not None else random.randint(0, 10)
        ),
        result=result,
        result_count=resolved_result_count,
        was_cancelled=(
            was_cancelled if was_cancelled is not None else random.choice([True, False])
        ),
        ground_truth=(
            ground_truth if ground_truth is not None else random.choice([True, False])
        ),
        created_at=created_at,
        updated_at=updated_at,
        lead_time=lead_time if lead_time is not None else random.uniform(0.1, 10.0),
        unique_id=unique_id if unique_id is not None else random_id(),
        bulk_created=(
            bulk_created if bulk_created is not None else random.choice([True, False])
        ),
        task=task if task is not None else random.randint(1, 1000),
        project=project if project is not None else random.randint(1, 100),
        updated_by=updated_by if updated_by is not None else random.randint(0, 10),
    )


def mother_task_data(
    image_url: Optional[str] = None,
    transcription: Optional[str] = None,
    transcription_length: Optional[int] = None,
) -> TaskData:
    image_url: str = (
        image_url if image_url is not None else f"img_{random.randint(1, 1000)}.png"
    )
    if transcription is None:
        length = (
            transcription_length
            if transcription_length is not None
            else int(np.random.randint(30, 50))
        )
        transcription: str = random_letters(length)
    return TaskData(image_url=image_url, transcription=transcription)


def mother_labelstudio_task(
    id: Optional[int] = None,
    inner_id: Optional[int] = None,
    file_upload: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    project: Optional[int] = None,
    updated_by: Optional[int] = None,
    data: Optional[TaskData] = None,
    annotations: Optional[list[RawAnnotation]] = None,
    drafts: Optional[list[Any]] = None,
    predictions: Optional[list[Any]] = None,
    meta: Optional[dict[Any, Any]] = None,
    total_annotations: Optional[int] = None,
    cancelled_annotations: Optional[int] = None,
    total_predictions: Optional[int] = None,
    comment_authors: Optional[list[Any]] = None,
    comment_count: Optional[int] = None,
    unresolved_comment_count: Optional[int] = None,
    task_data_kwargs: Optional[dict[str, Any]] = None,
    raw_annotation: Optional[RawAnnotation] = None,
    raw_annotation_kwargs: Optional[dict[str, Any]] = None,
    n_comment_authors: Optional[int] = None,
) -> LabelStudioTask:

    created_at: str = created_at if created_at is not None else "2026-01-01"
    updated_at: str = updated_at if updated_at is not None else "2026-01-01"
    if data is None:
        data: TaskData = mother_task_data(**(task_data_kwargs or {}))
    if annotations is None:
        annotations: list[RawAnnotation] = [
            (
                raw_annotation
                if raw_annotation is not None
                else mother_raw_annotation(**(raw_annotation_kwargs or {}))
            )
        ]
    if comment_authors is None:
        n_comment_authors: int = (
            n_comment_authors if n_comment_authors is not None else random.randint(0, 3)
        )
        comment_authors: list[str] = [
            f"user{random.randint(1, 10)}" for _ in range(n_comment_authors)
        ]
    return LabelStudioTask(
        id=id if id is not None else random.randint(1, 10000),
        inner_id=inner_id if inner_id is not None else random.randint(1, 10000),
        file_upload=(
            file_upload
            if file_upload is not None
            else f"file_{random.randint(1, 1000)}.png"
        ),
        created_at=created_at,
        updated_at=updated_at,
        project=project if project is not None else random.randint(1, 100),
        updated_by=updated_by if updated_by is not None else random.randint(0, 10),
        data=data,
        annotations=annotations,
        drafts=drafts if drafts is not None else [],
        predictions=predictions if predictions is not None else [],
        meta=meta if meta is not None else {},
        total_annotations=(
            total_annotations
            if total_annotations is not None
            else random.randint(1, 10)
        ),
        cancelled_annotations=(
            cancelled_annotations
            if cancelled_annotations is not None
            else random.randint(0, 5)
        ),
        total_predictions=(
            total_predictions if total_predictions is not None else random.randint(0, 5)
        ),
        comment_authors=comment_authors,
        comment_count=(
            comment_count if comment_count is not None else random.randint(0, 10)
        ),
        unresolved_comment_count=(
            unresolved_comment_count
            if unresolved_comment_count is not None
            else random.randint(0, 5)
        ),
    )


def mother_simplified_text_correction_value(
    text: Optional[str] = None,
    text_length: Optional[int] = None,
) -> SimplifiedTextCorrectionValue:
    if text is None:
        text: str = random_text(text_length)
    return SimplifiedTextCorrectionValue(text=[text])


def mother_simplified_text_correction_result(
    id: Optional[str] = None,
    to_name: Optional[str] = None,
    origin: Optional[str] = None,
    from_name: Optional[Literal["correction", "text_adapter"]] = None,
    value: Optional[SimplifiedTextCorrectionValue] = None,
    value_text: Optional[str] = None,
    value_text_length: Optional[int] = None,
) -> SimplifiedTextCorrectionResult:
    return SimplifiedTextCorrectionResult(
        id=id if id is not None else random_id(),
        to_name=to_name if to_name is not None else f"to_{random.randint(1, 10)}",
        type="textarea",
        origin=origin if origin is not None else f"origin_{random.randint(1, 5)}",
        from_name=from_name if from_name is not None else "correction",
        value=(
            value
            if value is not None
            else mother_simplified_text_correction_value(
                text=value_text,
                text_length=value_text_length,
            )
        ),
    )


def mother_simplified_annotation(
    id: Optional[int] = None,
    completed_by: Optional[int] = None,
    result: Optional[list[Any]] = None,
    result_count: Optional[int] = None,
    was_cancelled: Optional[bool] = None,
    ground_truth: Optional[bool] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    lead_time: Optional[float] = None,
    unique_id: Optional[str] = None,
    bulk_created: Optional[bool] = None,
    task: Optional[int] = None,
    project: Optional[int] = None,
    updated_by: Optional[int] = None,
    text_correction_result: Optional[SimplifiedTextCorrectionResult] = None,
    text_correction_result_kwargs: Optional[dict[str, Any]] = None,
    n_fragments: Optional[int] = None,
    n_paragraphs: Optional[int] = None,
    pair_images_and_texts: bool = True,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> SimplifiedAnnotation:

    created_at: str = created_at if created_at is not None else "2026-01-01"
    updated_at: str = updated_at if updated_at is not None else "2026-01-01"
    if result is None:
        resolved_n_paragraphs = n_paragraphs if n_paragraphs is not None else 1
        resolved_n_fragments = (
            n_fragments if n_fragments is not None else resolved_n_paragraphs
        )

        if resolved_n_paragraphs < 1:
            raise ValueError("n_paragraphs debe ser >= 1")
        if resolved_n_fragments < 1:
            raise ValueError("n_fragments debe ser >= 1")
        if resolved_n_paragraphs > resolved_n_fragments:
            raise ValueError("n_paragraphs no puede ser mayor que n_fragments")

        resolved_img_width = (
            image_width if image_width is not None else random.randint(50, 500)
        )
        resolved_img_height = (
            image_height if image_height is not None else random.randint(50, 500)
        )

        base = resolved_n_fragments // resolved_n_paragraphs
        remainder = resolved_n_fragments % resolved_n_paragraphs
        sizes = [
            base + (1 if i < remainder else 0) for i in range(resolved_n_paragraphs)
        ]

        gap = 100.0 / resolved_n_paragraphs
        box_width = max(0.5, min(20.0, gap * 0.7))
        box_height = 20.0

        generated_results: list[SimplifiedResultItem] = []
        for paragraph_index, paragraph_size in enumerate(sizes):
            base_x = paragraph_index * gap
            x = min(base_x + 0.5, max(0.0, 100.0 - box_width - 0.5))
            y = 10.0
            for fragment_index in range(paragraph_size):
                rect_id = random_id()
                text_id = random_id()
                local_dx = min(fragment_index * 0.1, box_width * 0.2)
                generated_results.append(
                    mother_rectangle_result(
                        id=rect_id,
                        original_width=resolved_img_width,
                        original_height=resolved_img_height,
                        image_rotation=0,
                        value_x=x + local_dx,
                        value_y=y,
                        value_width=box_width,
                        value_height=box_height,
                        value_rotation=0,
                    )
                )

                generated_results.append(
                    mother_simplified_text_correction_result(
                        id=text_id,
                        value=(
                            text_correction_result.value
                            if text_correction_result is not None
                            else None
                        ),
                        **(text_correction_result_kwargs or {}),
                    )
                )

                if pair_images_and_texts:
                    generated_results.append(
                        mother_relation_result(
                            from_id=rect_id,
                            to_id=text_id,
                            direction="right",
                        )
                    )

        result: list[SimplifiedResultItem] = generated_results
    return SimplifiedAnnotation(
        id=id if id is not None else random.randint(1, 10000),
        completed_by=(
            completed_by if completed_by is not None else random.randint(0, 10)
        ),
        result=result,
        result_count=result_count if result_count is not None else random.randint(1, 5),
        was_cancelled=(
            was_cancelled if was_cancelled is not None else random.choice([True, False])
        ),
        ground_truth=(
            ground_truth if ground_truth is not None else random.choice([True, False])
        ),
        created_at=created_at,
        updated_at=updated_at,
        lead_time=lead_time if lead_time is not None else random.uniform(0.1, 10.0),
        unique_id=unique_id if unique_id is not None else random_id(),
        bulk_created=(
            bulk_created if bulk_created is not None else random.choice([True, False])
        ),
        task=task if task is not None else random.randint(1, 1000),
        project=project if project is not None else random.randint(1, 100),
        updated_by=updated_by if updated_by is not None else random.randint(0, 10),
    )


def mother_simplified_task(
    id: Optional[int] = None,
    inner_id: Optional[int] = None,
    file_upload: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    project: Optional[int] = None,
    updated_by: Optional[int] = None,
    data: Optional[TaskData] = None,
    annotations: Optional[list[SimplifiedAnnotation]] = None,
    drafts: Optional[list[Any]] = None,
    predictions: Optional[list[Any]] = None,
    meta: Optional[dict[Any, Any]] = None,
    total_annotations: Optional[int] = None,
    cancelled_annotations: Optional[int] = None,
    total_predictions: Optional[int] = None,
    comment_authors: Optional[list[Any]] = None,
    comment_count: Optional[int] = None,
    unresolved_comment_count: Optional[int] = None,
    task_data_kwargs: Optional[dict[str, Any]] = None,
    simplified_annotation: Optional[SimplifiedAnnotation] = None,
    simplified_annotation_kwargs: Optional[dict[str, Any]] = None,
    n_comment_authors: Optional[int] = None,
) -> SimplifiedTask:

    created_at: str = created_at if created_at is not None else "2026-01-01"
    updated_at: str = updated_at if updated_at is not None else "2026-01-01"
    if data is None:
        data: TaskData = mother_task_data(**(task_data_kwargs or {}))
    if annotations is None:
        annotations: list[SimplifiedAnnotation] = [
            (
                simplified_annotation
                if simplified_annotation is not None
                else mother_simplified_annotation(
                    **(simplified_annotation_kwargs or {})
                )
            )
        ]
    if comment_authors is None:
        n_comment_authors: int = (
            n_comment_authors if n_comment_authors is not None else random.randint(0, 3)
        )
        comment_authors: list[str] = [
            f"user{random.randint(1, 10)}" for _ in range(n_comment_authors)
        ]
    return SimplifiedTask(
        id=id if id is not None else random.randint(1, 10000),
        inner_id=inner_id if inner_id is not None else random.randint(1, 10000),
        file_upload=(
            file_upload
            if file_upload is not None
            else f"file_{random.randint(1, 1000)}.png"
        ),
        created_at=created_at,
        updated_at=updated_at,
        project=project if project is not None else random.randint(1, 100),
        updated_by=updated_by if updated_by is not None else random.randint(0, 10),
        data=data,
        annotations=annotations,
        drafts=drafts if drafts is not None else [],
        predictions=predictions if predictions is not None else [],
        meta=meta if meta is not None else {},
        total_annotations=(
            total_annotations
            if total_annotations is not None
            else random.randint(1, 10)
        ),
        cancelled_annotations=(
            cancelled_annotations
            if cancelled_annotations is not None
            else random.randint(0, 5)
        ),
        total_predictions=(
            total_predictions if total_predictions is not None else random.randint(0, 5)
        ),
        comment_authors=comment_authors,
        comment_count=(
            comment_count if comment_count is not None else random.randint(0, 10)
        ),
        unresolved_comment_count=(
            unresolved_comment_count
            if unresolved_comment_count is not None
            else random.randint(0, 5)
        ),
    )


def mother_text_fragment(
    id: Optional[str] = None,
    text: Optional[str] = None,
    task_id: Optional[int] = None,
    text_length: Optional[int] = None,
    has_association: bool = True,
) -> TextFragment:
    if text is None:
        text: str = "".join(
            random.choices(
                random_id(),
                k=text_length if text_length is not None else random.randint(3, 10),
            )
        )
    text_fragment = TextFragment(
        id=id if id is not None else f"frag{random.randint(1, 1000)}",
        text=text,
        task_id=task_id if task_id is not None else random.randint(1, 1000),
    )
    if has_association:
        associated_box = mother_image_box(has_association=False)
        text_fragment.associate_box(associated_box)
        associated_box.associate_fragment(text_fragment)
    return text_fragment


def mother_image_box(
    id: Optional[str] = None,
    stroke_crop: Optional[Image.Image] = None,
    polygon: Optional[ShapelyPolygon] = None,
    rotation: Optional[float] = None,
    unrotated: Optional[bool] = None,
    task_id: Optional[int] = None,
    true_rectangle: Optional[bool] = None,
    n_points: Optional[int] = None,
    points: Optional[list[tuple[float, float]]] = None,
    pil_image_kwargs: Optional[dict[str, Any]] = None,
    has_association: bool = True,
) -> ImageBox:
    stroke_crop: Image.Image = (
        stroke_crop
        if stroke_crop is not None
        else mother_pil_image(**(pil_image_kwargs or {}))
    )
    if polygon is None:
        count = n_points if n_points is not None else random.randint(3, 8)
        points = (
            points
            if points is not None
            else [
                (random.uniform(0, 10), random.uniform(0, 10))
                for _ in range(max(3, count))
            ]
        )
        polygon: ShapelyPolygon = ShapelyPolygon(points)
    image_box = ImageBox(
        box_id=id if id is not None else f"box{random.randint(1, 1000)}",
        stroke_crop=stroke_crop,
        polygon=polygon,
        rotation=rotation if rotation is not None else random.uniform(0, 360),
        task_id=task_id if task_id is not None else random.randint(1, 1000),
        true_rectangle=(
            true_rectangle
            if true_rectangle is not None
            else random.choice([True, False])
        ),
    )
    if has_association:
        associated_fragment = mother_text_fragment(has_association=False)
        image_box.associate_fragment(associated_fragment)
        associated_fragment.associate_box(image_box)
    return image_box


def mother_paragraph(
    image_boxes: Optional[list[ImageBox]] = None,
    text_fragments: Optional[list[TextFragment]] = None,
    task_id: Optional[int] = None,
    index: Optional[int] = None,
    subgraph: Optional[dict[str, set[str]]] = None,
    fragment: Optional[TextFragment] = None,
    box: Optional[ImageBox] = None,
    text_fragment_kwargs: Optional[dict[str, Any]] = None,
    image_box_kwargs: Optional[dict[str, Any]] = None,
) -> Paragraph:
    if image_boxes is None and text_fragments is None:
        fragment: TextFragment = (
            fragment
            if fragment is not None
            else mother_text_fragment(**(text_fragment_kwargs or {}))
        )
        box: ImageBox = (
            box if box is not None else mother_image_box(**(image_box_kwargs or {}))
        )
        if not box.associated_fragments:
            box.associate_fragment(fragment)
        image_boxes = [box]
        text_fragments = []
        subgraph = subgraph if subgraph is not None else {box.box_id: set()}
    return Paragraph(
        image_boxes=image_boxes,
        text_fragments=text_fragments,
        task_id=task_id if task_id is not None else random.randint(1, 1000),
        index=index if index is not None else random.randint(0, 10),
        subgraph=subgraph,
    )


def mother_pil_image(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    color: Optional[tuple[int, int, int]] = None,
) -> Image.Image:
    width: int = width if width is not None else int(np.random.randint(5, 20))
    height: int = height if height is not None else int(np.random.randint(5, 20))

    if color is None:
        arr = np.random.randint(
            0,
            256,
            (height, width, 3),
            dtype=np.uint8,
        )
        return Image.fromarray(arr)

    return Image.new("RGB", (width, height), color)


def mother_annotated_page(
    ann: Optional[SimplifiedAnnotation] = None,
    img: Optional[Image.Image] = None,
    unrotate: Optional[bool] = False,
    usernames_labelstudio: Optional[list[str]] = None,
    simplified_annotation_kwargs: Optional[dict[str, Any]] = None,
    pil_image_kwargs: Optional[dict[str, Any]] = None,
    n_usernames: Optional[int] = None,
    n_fragments: Optional[int] = None,
    n_paragraphs: Optional[int] = None,
) -> AnnotatedPage:
    img: Image.Image = (
        img if img is not None else mother_pil_image(**(pil_image_kwargs or {}))
    )
    if ann is None:
        ann_kwargs = dict(simplified_annotation_kwargs or {})
        if n_fragments is not None:
            ann_kwargs["n_fragments"] = n_fragments
        if n_paragraphs is not None:
            ann_kwargs["n_paragraphs"] = n_paragraphs
        ann_kwargs.setdefault("image_width", img.width)
        ann_kwargs.setdefault("image_height", img.height)
        ann: SimplifiedAnnotation = mother_simplified_annotation(**ann_kwargs)
    if usernames_labelstudio is None:
        n_usernames: int = (
            n_usernames if n_usernames is not None else random.randint(1, 3)
        )
        usernames_labelstudio: list[str] = [
            f"user{random.randint(1, 10)}" for _ in range(n_usernames)
        ]
    return AnnotatedPage(
        ann,
        img,
        usernames_labelstudio=usernames_labelstudio,
    )
