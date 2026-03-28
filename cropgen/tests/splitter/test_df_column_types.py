from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.dataset_interface.DatasetInterface import DatasetInterface
from pytest import skip
import pandas as pd

_expected_columns = [
    "task",
    "paragraph",
    "order",
    "sindex",
    "text",
    "page",
    "crop_file",
    "background_color",
    "average_rotation",
]


def test_df_column_types(paths: PathBundle):
    dsi = DatasetInterface(paths)
    assert set(dsi.df.columns) == set(_expected_columns)

    assert dsi.df["task"].apply(lambda x: isinstance(x, int)).all()
    assert (
        dsi.df["paragraph"].apply(lambda x: isinstance(x, int) or (x == "full")).all()
    )
    assert (
        dsi.df["order"]
        .apply(lambda x: isinstance(x, int) or (x == "full") or (x == "paragraph"))
        .all()
    )
    assert dsi.df["sindex"].apply(lambda x: isinstance(x, int)).all()
    assert dsi.df["text"].apply(lambda x: isinstance(x, str)).all()
    assert dsi.df["crop_file"].apply(lambda x: isinstance(x, str)).all()
    assert dsi.df["page"].apply(lambda x: isinstance(x, int)).all()
    assert (
        dsi.df["background_color"]
        .apply(lambda x: isinstance(x, list) and all([isinstance(xi, int) for xi in x]))
        .all()
    )
    assert dsi.df["average_rotation"].apply(lambda x: isinstance(x, float)).all
