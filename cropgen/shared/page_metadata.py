from typing import Annotated
import json
from pydantic import BaseModel, AfterValidator, ValidationError
from pathlib import Path
from PIL import Image


def _validate_path_existance(path: Path):
    if not path.exists():
        return ValidationError("Failed path validation: path does not exist.")


def _load_path_content(path: Path):
    _validate_path_existance(path)
    return json.loads(path.read_text())


def _validate_polygon_path_content(polygons_path: Path):
    polygons_list = _load_path_content(polygons_path)

    if not isinstance(polygons_list, list):
        raise ValidationError(
            f"Failed polygon path validation: the outer element must be a list, found {type(polygons_list)}."
        )

    for polyElement in polygons_list:

        if not isinstance(polyElement, list):
            raise ValidationError(
                f"Failed polygon path validation: an element of the outer list is not a list itself, found {type(polyElement)}."
            )
        for xyElement in polyElement:
            if not isinstance(xyElement, tuple):
                raise ValidationError(
                    f"Failed polygon path validation: an element of the inner list is not a tuple, found {type(xyElement)}."
                )
            if not (len(xyElement) == 2):
                raise ValidationError(
                    f"Failed polygon path validation: one of the tuples in the inner list is of length {len(xyElement)} != 2."
                )

            if not (
                isinstance(xyElement[0], float) and isinstance(xyElement[1], float)
            ):
                raise ValidationError(
                    f"Failed polygon path validation: one of the tuples contains non-float objects: {[type(z) for z in xyElement]}"
                )


def _validate_transcriptions_path_content(transcriptions_path: Path):
    transcriptions = _load_path_content(transcriptions_path)

    if not isinstance(transcriptions, list):
        raise ValidationError(
            f"Failed transcriptions path validation: the outer element must be a list, found {type(transcriptions)}."
        )

    for transcription in transcriptions:
        if not isinstance(transcription, str):
            raise ValidationError(
                f"Failed transcrpitions path: expected all elements to be str, found {type(transcription)}"
            )


def _validate_rotations_path_content(rotations_path: Path):
    rotations = _load_path_content(rotations_path)

    if not isinstance(rotations, list):
        raise ValidationError(
            f"Failed rotations path validation: the outer element must be a list, found {type(rotations)}."
        )

    for transcription in rotations:
        if not isinstance(transcription, float):
            raise ValidationError(
                f"Failed rotations path: expected all elements to be float, found {type(transcription)}"
            )


def _validate_ids_path_content(ids_path: Path):
    ids = _load_path_content(ids_path)

    if not isinstance(ids, list):
        raise ValidationError(
            f"Failed ids path validation: the outer element must be a list, found {type(ids)}."
        )

    for individual_id in ids:
        if not isinstance(individual_id, str):
            raise ValidationError(
                f"Failed rotations path: expected all elements to be str, found {type(individual_id)}"
            )


def _validate_image_path(image_path: Path):
    stroke_path = (
        image_path.parents[1] / "strokes" / (image_path.stem + image_path.suffix)
    )
    background_path = image_path.parents[1] / (image_path.stem + image_path.suffix)
    if not image_path.exists():
        raise ValidationError("The raw image does not exist.")
    if not stroke_path.exists():
        raise ValidationError("The stroke image does not exist.")
    if not background_path.exists():
        raise ValidationError("The background image does not exist.")

    # too resource intensive for this general checks.
    # for path in [image_path, stroke_path, background_path]:
    #     try:
    #         Image.open(path).convert("L")
    #     except Exception as e:
    #         raise ValidationError(
    #             f"The image associated with path {path} failed to load and convert to grayscale with exception:\n{e}."
    #         )


class PageSampleMetadata(BaseModel):
    page: str
    task_id: int
    subindex: int
    completer: str
    updater: str
    ann_id: str
    order: int
    image_path: Annotated[Path, AfterValidator(_validate_image_path)]
    ids_path: Annotated[Path, AfterValidator(_validate_ids_path_content)]
    txt_path: Annotated[Path, AfterValidator(_validate_transcriptions_path_content)]
    polygons_path: Annotated[Path, AfterValidator(_validate_polygon_path_content)]
    rotations_path: Annotated[Path, AfterValidator(_validate_rotations_path_content)]
    polygons_are_in_percentage: bool
    source: str

    @property
    def transcriptions(self) -> list[str]:
        return _load_path_content(self.txt_path)

    @property
    def polygon_coords(self) -> list[list[tuple[float, float]]]:
        return _load_path_content(self.polygons_path)

    @property
    def rotations(self) -> list[float]:
        return _load_path_content(self.rotations_path)

    @property
    def ids(self) -> list[str]:
        return _load_path_content(self.ids_path)
