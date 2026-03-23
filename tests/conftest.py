# tests/conftest.py
from external_interfaces.LabelStudioInterface import LabelStudioInterface
import os
import pytest
from external_interfaces.download_from_bucket import download_all
from shared.PathBundle import PathBundle


@pytest.fixture(scope="session", autouse=True)
def prepare_data():
    paths = PathBundle()
    if (
        len(os.listdir(paths.images_path)) < 612
        or len(os.listdir(paths.transcriptions_path)) < 612
    ):
        download_all(paths)

    LabelStudioInterface.update_conditional(paths)
