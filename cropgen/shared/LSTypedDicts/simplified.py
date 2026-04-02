from pydantic import BaseModel, Field
from typing import Literal, Union, Optional, Any, List, Dict, Annotated

from cropgen.shared.LSTypedDicts.aggregates import TaskData
from cropgen.shared.LSTypedDicts.results import (
    BaseResult,
    RectangleResult,
    PolygonResult,
    RelationResult,
)


class SimplifiedTextCorrectionValue(BaseModel):
    text: List[str]


class SimplifiedTextCorrectionResult(BaseResult):
    from_name: Literal["correction", "text_adapter"]
    type: Literal["textarea"]
    value: SimplifiedTextCorrectionValue
    origin: Optional[str] = None  # sobreescribimos el base


SimplifiedResultItem = Annotated[
    Union[
        SimplifiedTextCorrectionResult, RectangleResult, PolygonResult, RelationResult
    ],
    Field(discriminator="type"),
]


class SimplifiedAnnotation(BaseModel):
    id: int
    completed_by: int
    result: List[SimplifiedResultItem]
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

    draft_created_at: Optional[str] = None
    import_id: Optional[int] = None
    last_action: Optional[Any] = None
    last_created_by: Optional[Any] = None
    parent_annotation: Optional[int] = None
    parent_prediction: Optional[Any] = None
    prediction: Dict = {}


class SimplifiedTask(BaseModel):
    id: int
    inner_id: int
    file_upload: str
    created_at: str
    updated_at: str
    project: int
    updated_by: int

    data: TaskData
    annotations: List[SimplifiedAnnotation]
    drafts: List[Any]
    predictions: List[Any]
    meta: Dict

    total_annotations: int
    cancelled_annotations: int
    total_predictions: int

    comment_authors: List[Any]
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Optional[Any] = None
