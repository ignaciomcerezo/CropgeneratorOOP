from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface

_expected_columns = [
    "task",
    "id",
    "paragraph",
    "order",
    "sindex",
    "text",
    "page",
    "crop_file",
    "background_color",
    "average_rotation",
    "has_enough_context",
    "is_letter",
]


def test_df_column_types(paths: PathBundle):
    pdi = PairsDataInterface(paths)
    assert set(pdi.df.columns) == set(_expected_columns)

    assert pdi.df["id"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["task"].apply(lambda x: isinstance(x, int)).all()
    assert (
        pdi.df["paragraph"].apply(lambda x: isinstance(x, int) or (x == "full")).all()
    )
    assert (
        pdi.df["order"]
        .apply(lambda x: isinstance(x, int) or (x == "full") or (x == "paragraph"))
        .all()
    )
    assert pdi.df["sindex"].apply(lambda x: isinstance(x, int)).all()
    assert pdi.df["text"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["crop_file"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["page"].apply(lambda x: isinstance(x, str)).all()
    assert (
        pdi.df["background_color"]
        .apply(lambda x: isinstance(x, list) and all([isinstance(xi, int) for xi in x]))
        .all()
    )
    assert pdi.df["average_rotation"].apply(lambda x: isinstance(x, float)).all
    assert pdi.df["has_enough_context"].apply(lambda x: isinstance(x, bool)).all
    assert pdi.df["is_letter"].apply(lambda x: isinstance(x, bool)).all
