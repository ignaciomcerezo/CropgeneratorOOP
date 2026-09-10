from pydantic import BaseModel
from typing import Union, List


class TextRegionValue(BaseModel):
    start: int
    end: int
    text: str
    labels: List[str]


class TextCorrectionValue(BaseModel):
    start: int
    end: int
    text: List[str]


class RectangleValue(BaseModel):
    x: Union[float, int]
    y: Union[float, int]
    width: Union[float, int]
    height: Union[float, int]
    rotation: Union[float, int]
    rectanglelabels: List[str]


class PolygonValue(BaseModel):
    points: List[List[Union[float, int]]]
    closed: bool
    polygonlabels: List[str]
