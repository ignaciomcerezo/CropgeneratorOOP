import pytest
from pathlib import Path
from cropgen.external_interfaces.online_bucket_interface import OnlineBucketInterface
from cropgen.shared.path_bundle import PathBundle


def test_obi_downloads_are_explicit(monkeypatch, paths: PathBundle):
    aimed_numer_of_calls = 5

    def fake_download(*args, **kwargs):
        if hasattr(fake_download, "download_calls"):
            fake_download.download_calls += 1
        else:
            setattr(fake_download, "download_calls", 1)
        return []

    def fake_test_connection(*args, **kwargs):
        return True

    monkeypatch.setattr(OnlineBucketInterface, "_download_image", fake_download)
    monkeypatch.setattr(
        OnlineBucketInterface, "test_connection_successful", fake_test_connection
    )
    monkeypatch.setattr(
        OnlineBucketInterface,
        "_compute_pending_objects",
        lambda *args, **kwargs: {str(x): str(x) for x in range(aimed_numer_of_calls)},
    )
    obi = OnlineBucketInterface("https://example.test/bucket/")
    assert getattr(fake_download, "download_calls", 0) == 0

    obi.setup(paths)
    assert getattr(fake_download, "download_calls") == aimed_numer_of_calls


@pytest.mark.parametrize("page_name", ["015", "154"])
def test_download_single_image(
    page_name: str, paths: PathBundle, obi: OnlineBucketInterface
):
    paths.remove_downloaded_image(page_name)

    obi.setup(paths)

    image_path = paths.get_raw_image_path(page_name)

    assert Path(image_path).exists(), f"La imagen no fue descargada: {image_path}"


def test_check_updates_and_update(paths: PathBundle, obi: OnlineBucketInterface):
    page_name = "015"
    paths.remove_downloaded_image(page_name)
    pendientes = list(obi._compute_pending_objects(paths).keys())
    assert page_name in pendientes or len(pendientes) == 0
    obi.setup(paths)
    pendientes = list(obi._compute_pending_objects(paths).keys())
    assert len(pendientes) == 0


def test_from_env(paths: PathBundle, bucket_url):
    obi2 = OnlineBucketInterface.from_env(paths)
    assert isinstance(obi2, OnlineBucketInterface)
    assert hasattr(obi2, "bucket_url")


def test_no_download_when_up_to_date(paths: PathBundle, obi: OnlineBucketInterface):

    obi.setup(paths)
    pendientes = list(obi._compute_pending_objects(paths).keys())
    assert pendientes == []
