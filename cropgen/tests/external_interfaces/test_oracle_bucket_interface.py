import pytest
from pathlib import Path
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.shared.PathBundle import PathBundle


@pytest.mark.parametrize("page_name", ["015", "154"])
def test_download_single_image_and_transcription(
    page_name: str, paths: PathBundle, obi: OracleBucketInterface
):
    paths.remove_downloaded_image_and_transcription(page_name)

    obi.update()

    image_path = paths.get_image_path(page_name)
    transcription_path = paths.get_transcription_path(page_name)

    assert Path(image_path).exists(), f"La imagen no fue descargada: {image_path}"
    assert Path(
        transcription_path
    ).exists(), f"La transcripción no fue descargada: {transcription_path}"


def test_check_updates_and_update(paths: PathBundle, obi: OracleBucketInterface):
    page_name = "015"
    paths.remove_downloaded_image_and_transcription(page_name)
    pendientes = obi.check_updates()
    assert isinstance(pendientes, list)
    assert page_name in pendientes or len(pendientes) == 0
    descargadas = obi.update()
    assert isinstance(descargadas, list)
    assert page_name in descargadas or len(descargadas) == 0
    descargadas2 = obi.update()
    assert descargadas2 == []


def test_from_env(paths: PathBundle, bucket_url):
    obi2 = OracleBucketInterface.from_env(paths)
    assert isinstance(obi2, OracleBucketInterface)
    assert hasattr(obi2, "bucket_url")


def test_no_download_when_up_to_date(paths: PathBundle, obi: OracleBucketInterface):
    pendientes = obi.check_updates()
    if pendientes:
        obi.update()
    descargadas = obi.update()
    assert descargadas == []
