# tests/conftest.py
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pytest
import os
from cropgen.shared.PathBundle import PathBundle
from dotenv import load_dotenv
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
import multiprocessing
from typing import Optional


@pytest.fixture(scope="session")
def paths() -> PathBundle:
    return PathBundle()


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
def lsi(paths, ls_token, ls_url) -> LabelStudioInterface:
    return LabelStudioInterface(paths)


@pytest.fixture(scope="session")
def obi(paths, bucket_url):
    obi = OracleBucketInterface(paths, bucket_url)
    return obi


@pytest.fixture(scope="session", autouse=True)
def prepare_data(paths, obi, ls_url, ls_token):
    load_dotenv()

    obi.update()
    LabelStudioInterface.update_conditional(paths, ls_url, ls_token)

    yield

    # paths.remove_all_files()


# TODO: implement a too_much_connectivity test (connectivity of a node > 0.3) for example


@pytest.fixture
def five_letter_task_numbers() -> list[int]:
    return [280, 293, 298, 305]


@pytest.fixture
def five_laloma_task_numbers() -> list[int]:
    return [1, 101, 143, 465, 526]


@pytest.fixture
def two_paragraph_laloma() -> list[int]:
    return [364, 365, 390, 460]


@pytest.fixture
def three_paragraph_laloma() -> list[int]:
    return [355, 366, 463]


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


@pytest.fixture
def pdi(paths) -> Optional[PairsDataInterface]:
    """Si pairls.jsonl no existe, devuelve None"""
    if not paths.json_filepath.exists():
        return None
    return PairsDataInterface(paths)
