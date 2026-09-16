
from pydantic import BaseModel


class TextRegionValue(BaseModel):
    start: int
    end: int
    text: str
    labels: list[str]


class TextCorrectionValue(BaseModel):
    start: int
    end: int
    text: list[str]


class RectangleValue(BaseModel):
    x: float | int
    y: float | int
    width: float | int
    height: float | int
    rotation: float | int
    rectanglelabels: list[str]


class PolygonValue(BaseModel):
    points: list[list[float | int]]
    closed: bool
    polygonlabels: list[str]
