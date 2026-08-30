# tests/conftest.py
from cropgen.shared.image_processing import separate_background_and_stroke
from cropgen.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
import numpy as np
from tqdm.auto import tqdm
from tkinter import Label
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import pytest
from dotenv import load_dotenv
from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.external_interfaces.online_bucket_interface import OnlineBucketInterface
from cropgen.shared.path_bundle import PathBundle
from requests.exceptions import ConnectionError
from PIL import Image


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
        lsi.setup()
    except ConnectionError:
        print("Connection errored for LabelStudioInterface, going offline.")
        lsi = LabelStudioInterface(paths, ls_url, ls_token, online=False)
        lsi.setup()
    return lsi


@pytest.fixture(scope="session")
def obi(paths: PathBundle, bucket_url: str) -> OnlineBucketInterface:
    try:
        obi = OnlineBucketInterface(paths, bucket_url)
    except ConnectionError:
        print("Connection errored for , going offline.")
        obi = OnlineBucketInterface(paths, bucket_url, online=False)
    return obi


@pytest.fixture(scope="session", autouse=True)
def prepare_data(
    paths: PathBundle, obi: OnlineBucketInterface, lsi: LabelStudioInterface
):

    for task in tqdm(
        [
            task
            for task in lsi.simplified_tasks
            if not paths.has_processed_images(lsi.get_image_stem_from_task(task))
        ],
        desc="Stroke/background separation.",
    ):
        raw_image_path = lsi.get_raw_image_path_from_task(task)

        if raw_image_path is None:
            raise ValueError(
                f"Internal error. Did not download all appropriate images: {raw_image_path}"
            )

        raw_image = Image.open(raw_image_path)

        background, stroke = separate_background_and_stroke(
            raw_image,
            out_longest_side=DATASET_LONGEST_SIZE_PX,
            processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
        )
        stroke.save(paths.stroke_images_path / f"{raw_image_path.stem}.png")
        background.save(paths.background_images_path / f"{raw_image_path.stem}.png")

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


def _fake_synthetic_manuscript(*args, **kwargs):
    return Image.Image(), []


@pytest.fixture
def patch_synthetic_manuscript(monkeypatch):
    print("Patching synthetic image generation.")
    monkeypatch.setattr(
        AnnotatedPage, "synthetic_manuscript", _fake_synthetic_manuscript
    )


@pytest.fixture(autouse=True)
def patch_image_open(mode="r", formats=None):
    return Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L")
