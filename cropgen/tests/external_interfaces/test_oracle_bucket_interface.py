import pytest
from pathlib import Path
from cropgen.external_interfaces.online_bucket_interface import OnlineBucketInterface
from cropgen.shared.PathBundle import PathBundle


@pytest.mark.parametrize("page_name", ["015", "154"])
def test_download_single_image(
    page_name: str, paths: PathBundle, obi: OnlineBucketInterface
):
    paths.remove_downloaded_image(page_name)

    obi.update()

    image_path = paths.get_raw_image_path(page_name)

    assert Path(image_path).exists(), f"La imagen no fue descargada: {image_path}"


def test_check_updates_and_update(paths: PathBundle, obi: OnlineBucketInterface):
    page_name = "015"
    paths.remove_downloaded_image(page_name)
    pendientes = obi.check_updates()
    assert isinstance(pendientes, list)
    assert page_name in pendientes or len(pendientes) == 0
    descargadas = obi.update()
    assert isinstance(descargadas, list)
    assert page_name in descargadas or len(descargadas) == 0
    descargadas2 = obi.update()
    assert descargadas2 == []


def test_from_env(paths: PathBundle, bucket_url):
    obi2 = OnlineBucketInterface.from_env(paths)
    assert isinstance(obi2, OnlineBucketInterface)
    assert hasattr(obi2, "bucket_url")


def test_no_download_when_up_to_date(paths: PathBundle, obi: OnlineBucketInterface):
    pendientes = obi.check_updates()
    if pendientes:
        obi.update()
    descargadas = obi.update()
    assert descargadas == []
