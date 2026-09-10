from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from cropgen.loading.external_interfaces.label_studio.ls_typed_dicts.aggregates import (
    TaskData,
)
from cropgen.loading.external_interfaces.label_studio.ls_typed_dicts.results import (
    BaseResult,
    PolygonResult,
    RectangleResult,
    RelationResult,
)


class SimplifiedTextCorrectionValue(BaseModel):
    text: list[str]


class SimplifiedTextCorrectionResult(BaseResult):
    from_name: Literal["correction", "text_adapter"]
    type: Literal["textarea"]
    value: SimplifiedTextCorrectionValue
    origin: str | None = None  # sobreescribimos el base


SimplifiedResultItem = Annotated[
    SimplifiedTextCorrectionResult | RectangleResult | PolygonResult | RelationResult,
    Field(discriminator="type"),
]


class SimplifiedAnnotation(BaseModel):
    id: int
    completed_by: int
    result: list[SimplifiedResultItem]
    result_count: int
    was_cancelled: bool
    ground_truth: bool
    created_at: str
    updated_at: str
    lead_time: float
    unique_id: str
    bulk_created: bool
    task: int
    project: int
    updated_by: int

    draft_created_at: str | None = None
    import_id: int | None = None
    last_action: Any | None = None
    last_created_by: Any | None = None
    parent_annotation: int | None = None
    parent_prediction: Any | None = None
    prediction: dict = {}


class SimplifiedTask(BaseModel):
    id: int
    inner_id: int
    file_upload: str
    created_at: str
    updated_at: str
    project: int
    updated_by: int

    data: TaskData
    annotations: list[SimplifiedAnnotation]
    drafts: list[Any]
    predictions: list[Any]
    meta: dict

    total_annotations: int
    cancelled_annotations: int
    total_predictions: int

    comment_authors: list[Any]
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Any | None = None
