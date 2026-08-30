import pytest
from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.shared.path_bundle import PathBundle
from cropgen.external_interfaces.label_studio.ls_typed_dicts import SimplifiedAnnotation


def test_lsi_raw_and_simplified_tasks(lsi: LabelStudioInterface):
    assert isinstance(lsi.raw_tasks, list)
    assert isinstance(lsi.simplified_tasks, list)
    assert len(lsi.raw_tasks) > 0, "No hay tareas raw en el export."
    assert len(lsi.simplified_tasks) > 0, "No hay tareas simplificadas."


def test_lsi_users(lsi: LabelStudioInterface):
    users = lsi.users()
    assert isinstance(users, list)


def test_lsi_setup_is_explicit(monkeypatch, tmp_path):
    paths = PathBundle(tmp_path)
    paths.raw_export_filepath.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_export_filepath.write_text("[]", encoding="utf-8")
    paths.simplified_filepath.write_text("[]", encoding="utf-8")
    calls = []

    def fake_fetch_and_simplify(self):
        calls.append("fetch")
        self.usernames = ["user-1"]
        self.paths.raw_export_filepath.write_text("[]", encoding="utf-8")
        self.paths.simplified_filepath.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(LabelStudioInterface, "fetch_and_simplify", fake_fetch_and_simplify)

    lsi = LabelStudioInterface(paths, "https://example.test", "token", online=True)
    assert calls == []

    lsi.setup()
    assert calls == ["fetch"]


def test_lsi_annotations(lsi: LabelStudioInterface):
    annotations = lsi.annotations
    assert isinstance(annotations, list)
    if annotations:
        assert isinstance(annotations[0], SimplifiedAnnotation)


@pytest.mark.parametrize("index", [0, 1, "0", "1"])
def test_lsi_getitem(lsi: LabelStudioInterface, index):
    items = lsi[index]
    assert isinstance(items, list)
