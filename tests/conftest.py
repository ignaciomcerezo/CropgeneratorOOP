# tests/conftest.py
from labelstudio.LabelStudioInterface import LabelStudioInterface
import os
import pytest
from downloaders.download_from_bucket import download_all
from kaggle_integration.PathBundle import PathBundle


@pytest.fixture(scope="session", autouse=True)
def prepare_data():
    paths = PathBundle()
    if (
        len(os.listdir(paths.images_path)) < 612
        or len(os.listdir(paths.transcriptions_path)) < 612
    ):
        download_all(paths)

    LabelStudioInterface.update_conditional(paths)
