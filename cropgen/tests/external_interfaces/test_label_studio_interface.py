import pytest
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle


def test_lsi_raw_and_simplified_tasks(lsi: LabelStudioInterface):
    assert isinstance(lsi.raw_tasks, list)
    assert isinstance(lsi.simplified_tasks, list)
    assert len(lsi.raw_tasks) > 0, "No hay tareas raw en el export."
    assert len(lsi.simplified_tasks) > 0, "No hay tareas simplificadas."


def test_lsi_users(lsi: LabelStudioInterface):
    users = lsi.users()
    assert isinstance(users, list)


def test_lsi_annotations(lsi: LabelStudioInterface):
    annotations = lsi.annotations
    assert isinstance(annotations, list)
    if annotations:
        assert isinstance(annotations[0], dict)


@pytest.mark.parametrize("index", [0, 1, "0", "1"])
def test_lsi_getitem(lsi: LabelStudioInterface, index):
    items = lsi[index]
    assert isinstance(items, list)


def test_lsi_save_raw_and_simplified(lsi: LabelStudioInterface, paths: PathBundle):
    paths.clean_export_folder()
    assert not paths.raw_export_filepath.exists()
    assert not paths.simplified_filepath.exists()
    lsi.save_raw_export()
    lsi.save_simplified_export()
    assert paths.raw_export_filepath.exists()
    assert paths.simplified_filepath.exists()
