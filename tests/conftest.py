# tests/conftest.py
from pathlib import Path
import os
import pytest
from downloaders.download_from_bucket import download_all_images


@pytest.fixture(scope="session", autouse=True)
def prepare_data():
    base = Path(os.getcwd()) / "data" / "input"
    imagenes = base / "images"
    transcripciones = base / "transcripciones"

    if len(os.listdir(imagenes)) < 612 or len(os.listdir(transcripciones)) < 612:
        download_all_images()
