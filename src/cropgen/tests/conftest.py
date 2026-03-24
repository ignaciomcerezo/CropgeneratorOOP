# tests/conftest.py
from src.cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
import pytest
import os
from src.cropgen.shared.PathBundle import PathBundle
from dotenv import load_dotenv
from src.cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface


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
