# tests/conftest.py
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from external_interfaces.OracleBucketInterface import OracleBucketInterface
import pytest
from shared.PathBundle import PathBundle


@pytest.fixture(scope="session", autouse=True)
def prepare_data():
    paths = PathBundle()

    OracleBucketInterface.from_env(paths).update()
    LabelStudioInterface.update_conditional(paths)

    yield

    paths.remove_all_files()
