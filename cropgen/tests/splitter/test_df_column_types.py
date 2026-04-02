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
    "context_length_chars",
    "context_length_words",
    "is_letter",
]


def _has_correct_background_format(value: list[int]):
    assert isinstance(value, list)
    assert len(value) == 3
    assert all([isinstance(xi, int) for xi in value])
    return True


def test_df_column_types(paths: PathBundle):
    pdi = PairsDataInterface(paths)
    assert set(pdi.df.columns) == set(_expected_columns)

    assert pdi.df["id"].apply(lambda x: isinstance(x, int)).all()
    assert pdi.df["page"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["page"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["sindex"].apply(lambda x: isinstance(x, int)).all()
    assert pdi.df["text"].apply(lambda x: isinstance(x, str)).all()
    assert pdi.df["crop_file"].apply(lambda x: isinstance(x, str)).all()
    assert (
        pdi.df["paragraph"].apply(lambda x: isinstance(x, int) or (x == "full")).all()
    )
    assert (
        pdi.df["order"]
        .apply(lambda x: isinstance(x, int) or (x == "full") or (x == "paragraph"))
        .all()
    )
    assert pdi.df["background_color"].apply(_has_correct_background_format).all()
    assert pdi.df["average_rotation"].apply(lambda x: isinstance(x, float)).all
    assert pdi.df["context_length_chars"].apply(lambda x: isinstance(x, int)).all
    assert pdi.df["context_length_words"].apply(lambda x: isinstance(x, int)).all
    assert pdi.df["is_letter"].apply(lambda x: isinstance(x, bool)).all
