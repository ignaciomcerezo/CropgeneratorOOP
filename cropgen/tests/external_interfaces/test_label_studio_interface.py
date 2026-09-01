import pytest
from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.shared.path_bundle import PathBundle
from cropgen.external_interfaces.label_studio.ls_typed_dicts import SimplifiedAnnotation


def test_lsi_users(lsi: LabelStudioInterface):
    users = lsi.users()
    assert isinstance(users, list)


# TODO: reimplement tests after refactoring of LSI
