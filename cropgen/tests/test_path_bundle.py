from cropgen.shared.PathBundle import PathBundle


def test_pathbundle_paths_exist(paths: PathBundle):
    assert paths.data_in_path.exists() and paths.data_in_path.is_dir()
    assert paths.images_path.exists() and paths.images_path.is_dir()
    assert paths.transcriptions_path.exists() and paths.transcriptions_path.is_dir()
    assert paths.exports_path.exists() and paths.exports_path.is_dir()
    assert paths.data_out_path.exists() and paths.data_out_path.is_dir()
    assert paths.crops_path.exists() and paths.crops_path.is_dir()


def test_pathbundle_get_image_and_transcription_path(paths: PathBundle):
    assert paths.get_image_path("015").exists()
    assert paths.get_transcription_path("015").exists()
    assert paths.get_image_path(15).exists()
    assert paths.get_transcription_path(15).exists()


def test_pathbundle_normalize_name():
    assert PathBundle._normalize_page_name(5) == "005"
    assert PathBundle._normalize_page_name("5") == "005"
    assert PathBundle._normalize_page_name("015") == "015"
    assert PathBundle._normalize_page_name(123) == "123"


def test_pathbundle_get_image_path_from_task(paths: PathBundle):
    task = {"data": {"image": "015.png"}}
    img_path = paths.get_image_path_from_task(task)
    assert img_path is not None and img_path.exists()
    task2 = {"data": {}}
    assert paths.get_image_path_from_task(task2) is None


def test_pathbundle_remove_and_clean(paths: PathBundle):
    temp_img = paths.images_path / "temp_test.png"
    temp_txt = paths.transcriptions_path / "temp_test.txt"
    temp_img.write_bytes(b"test")
    temp_txt.write_text("test", encoding="utf-8")
    assert temp_img.exists() and temp_txt.exists()
    paths.remove_downloaded_image_and_transcription("temp_test.png")
    assert not temp_img.exists() and not temp_txt.exists()
