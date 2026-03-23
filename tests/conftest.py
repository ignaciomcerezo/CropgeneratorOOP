# tests/conftest.py
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from external_interfaces.OracleBucketInterface import OracleBucketInterface
import pytest
import os
from shared.PathBundle import PathBundle
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def prepare_data():
    load_dotenv()
    paths = PathBundle()

    OracleBucketInterface.from_env(paths).update()
    LabelStudioInterface.update_conditional(
        paths, os.getenv("LS_URL"), os.getenv("LS_TOKEN")
    )

    yield

    paths.remove_all_files()


@pytest.fixture
def ls_token():
    return os.getenv("ls_token")


@pytest.fixture
def ls_url():
    return os.getenv("LS_URL")
