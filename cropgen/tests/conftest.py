# tests/conftest.py
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
import pytest
import os
from cropgen.shared.PathBundle import PathBundle
from dotenv import load_dotenv
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
import multiprocessing


@pytest.fixture(scope="session")
def paths():
    return PathBundle()


@pytest.fixture(scope="session")
def ls_token():
    return os.getenv("LS_TOKEN")


@pytest.fixture(scope="session")
def ls_url():
    return os.getenv("LS_URL")


@pytest.fixture(scope="session")
def lsi(paths, ls_token, ls_url):
    return LabelStudioInterface(paths)


@pytest.fixture(scope="session", autouse=True)
def prepare_data(paths, ls_url, ls_token):
    load_dotenv()

    OracleBucketInterface.from_env(paths).update()
    LabelStudioInterface.update_conditional(paths, ls_url, ls_token)

    yield

    # paths.remove_all_files()


# TODO: implement a too_much_connectivity test (connectivity of a node > 0.3) for example


@pytest.fixture
def five_letter_task_numbers():
    return [280, 293, 298, 305]


@pytest.fixture
def five_laloma_task_numbers():
    return [1, 101, 143, 465, 526]


@pytest.fixture
def two_paragraph_laloma():
    return [364, 365, 390, 460]


@pytest.fixture
def three_paragraph_laloma():
    return [355, 366, 463]


@pytest.fixture
def tasks_with_margin_separate_annotation():
    return [358, 363, 367, 408, 456, 457, 362]


@pytest.fixture
def task_macedonia(
    five_laloma_task_numbers,
    five_letter_task_numbers,
    two_paragraph_laloma,
    three_paragraph_laloma,
    tasks_with_margin_separate_annotation,
):
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
