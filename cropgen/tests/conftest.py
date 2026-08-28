# tests/conftest.py
from tkinter import Label
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import pytest
from dotenv import load_dotenv
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.shared.PathBundle import PathBundle
from requests.exceptions import ConnectionError


@pytest.fixture(scope="session")
def paths() -> PathBundle:
    return PathBundle(Path(os.getcwd()))


@pytest.fixture(scope="session")
def ls_token() -> str:
    load_dotenv()
    res = os.getenv("LS_TOKEN")
    assert res is not None
    return res


@pytest.fixture(scope="session")
def ls_url() -> str:
    load_dotenv()
    res = os.getenv("LS_URL")
    assert res is not None
    return res


@pytest.fixture(scope="session")
def bucket_url() -> str:
    load_dotenv()
    res = os.getenv("BUCKET_URL")
    assert res is not None
    return res


@pytest.fixture(scope="session")
def lsi(paths: PathBundle, ls_token, ls_url) -> LabelStudioInterface:
    try:
        lsi = LabelStudioInterface(paths, ls_url, ls_token)
    except ConnectionError:
        print("Connection errored for LabelStudioInterface, going offline.")
        lsi = LabelStudioInterface(paths, ls_url, ls_token, online=False)

    paths.lsi = lsi
    return lsi


@pytest.fixture(scope="session")
def obi(paths: PathBundle, bucket_url: str) -> OracleBucketInterface:
    try:
        obi = OracleBucketInterface(paths, bucket_url)
    except ConnectionError:
        print("Connection errored for OracleBucketInterface, going offline.")
        obi = OracleBucketInterface(paths, bucket_url, online=False)

    paths.obi = obi
    return obi


@pytest.fixture(scope="session", autouse=True)
def prepare_data(
    paths: PathBundle, obi: OracleBucketInterface, lsi: LabelStudioInterface
):

    yield


@pytest.fixture
def five_letter_task_numbers() -> list[int]:
    return [280, 293, 298, 305]


@pytest.fixture
def five_laloma_task_numbers() -> list[int]:
    return [1, 101, 138, 465, 526]


@pytest.fixture
def two_paragraph_laloma() -> list[int]:
    return [364, 365, 390, 460]


@pytest.fixture
def three_paragraph_laloma() -> list[int]:
    return [128, 132, 140, 366, 463]


@pytest.fixture
def tasks_with_margin_separate_annotation() -> list[int]:
    return [358, 363, 367, 408, 456, 457, 362]


@pytest.fixture
def task_macedonia(
    five_laloma_task_numbers,
    five_letter_task_numbers,
    two_paragraph_laloma,
    three_paragraph_laloma,
    tasks_with_margin_separate_annotation,
) -> list[int]:
    return (
        five_laloma_task_numbers
        + five_letter_task_numbers
        + two_paragraph_laloma
        + three_paragraph_laloma
        + tasks_with_margin_separate_annotation
    )


@pytest.fixture(autouse=True, scope="session")
def set_multiprocessing_start_method():
    multiprocessing.set_start_method("spawn", force=True)


import cropgen.shared.image_processing as imgproc
from cropgen.processing.annotated_page import AnnotatedPage


def _fake_separate_background_and_stroke(
    img, *args, **kwargs
) -> tuple[Image.Image, Image.Image]:
    return img, img


def _fake_synthetic_manuscript(*args, **kwargs):
    return Image.Image(), []


@pytest.fixture(autouse=True)
def patch_image_processing(monkeypatch):
    print("Patching image processing.")
    monkeypatch.setattr(
        imgproc,
        "separate_background_and_stroke",
        _fake_separate_background_and_stroke,
    )


@pytest.fixture
def patch_synthetic_manuscript(monkeypatch):
    print("Patching synthetic image generation.")
    monkeypatch.setattr(
        AnnotatedPage, "synthetic_manuscript", _fake_synthetic_manuscript
    )
