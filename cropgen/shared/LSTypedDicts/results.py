from pydantic import BaseModel
from typing import Literal
from cropgen.shared.LSTypedDicts.values import (
    TextRegionValue,
    TextCorrectionValue,
    RectangleValue,
    PolygonValue,
)


class BaseResult(BaseModel):
    id: str
    to_name: str
    type: str
    origin: str


class ImageBaseResult(BaseResult):
    original_width: int
    original_height: int
    image_rotation: int | float


class TextRegionResult(BaseResult):
    from_name: Literal["txt_spans"]
    type: Literal["labels", "hypertextlabels"]
    value: TextRegionValue


class TextCorrectionResult(BaseResult):
    from_name: Literal["correction"]
    type: Literal["textarea"]
    value: TextCorrectionValue


class RectangleResult(ImageBaseResult):
    from_name: Literal["img_regions"]
    type: Literal["rectanglelabels"]
    value: RectangleValue


class PolygonResult(ImageBaseResult):
    from_name: Literal["img_polygons"]
    type: Literal["polygonlabels"]
    value: PolygonValue


class RelationResult(BaseModel):
    from_id: str
    to_id: str
    type: Literal["relation"]
    direction: Literal["right", "left", "bi"]
