import pytest
import pandas as pd

from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.splitter.statistical_report import (
    PairsStatisticalData,
    count_tokens,
    math_percentage,
)

NUMERIC_DESC_COLUMNS = ["mean", "min", "max", "std"]


def test_count_tokens_con_comandos_latex_y_puntuacion():
    text = r"\alpha + beta, gamma."
    assert count_tokens(text) == 6


def test_math_percentage_texto_llano_y_vacio():
    assert math_percentage("solo texto") == 0.0
    assert math_percentage("") == 0.0


def test_math_percentage_calculo_mixto_y_dolares_no_balanceados():
    # math: x + y (3 tokens), texto: hola + mundo (2 tokens)
    assert math_percentage("hola $x+y$ mundo") == 3 / 5
    assert math_percentage("hola $x+y mundo") == -1.0


def test_pairs_statistical_data_agrega_columnas_derivadas(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    stats = PairsStatisticalData(pdi)

    assert "text_length" in stats.df.columns
    assert "math_percentage" in stats.df.columns

    expected_text_length = pdi.df["text"].apply(len)
    expected_math_percentage = pdi.df["text"].apply(math_percentage)

    pd.testing.assert_series_equal(
        stats.df["text_length"], expected_text_length, check_names=False
    )
    pd.testing.assert_series_equal(
        stats.df["math_percentage"], expected_math_percentage, check_names=False
    )


def test_pairs_statistical_data_genera_resumenes(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    stats = PairsStatisticalData(pdi)

    assert list(stats.sindex.columns) == NUMERIC_DESC_COLUMNS
    assert stats.sindex.iloc[0]["mean"] == pytest.approx(pdi.df.sindex.mean())
    assert stats.text_length.iloc[0]["min"] == pdi.df.text.apply(len).min()

    # describe() de categóricas produce count/unique/top/freq
    assert "order" in stats.order.columns
    assert stats.order.loc["count", "order"] == len(pdi.df)


def test_stratify_one_by_other_numerica(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    df = pdi.df
    result = PairsStatisticalData.stratify_one_by_other(df, "sindex", "paragraph")

    expected_strata = sorted([str(x) for x in pd.unique(df["paragraph"])])
    assert set(result.index) == set(expected_strata)

    for strata_value in expected_strata:
        expected_mean = df[df["paragraph"] == strata_value]["sindex"].mean()
        got_mean = result.loc[strata_value, "mean"]
        if pd.isna(expected_mean):
            assert pd.isna(got_mean)
        else:
            assert got_mean == pytest.approx(expected_mean)


def test_stratify_one_by_other_categorica(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    result = PairsStatisticalData.stratify_one_by_other(pdi.df, "order", "paragraph")

    expected_strata = sorted([str(x) for x in pd.unique(pdi.df["paragraph"])])
    assert set(result.index) == set(expected_strata)
    assert "order" in result.columns


def test_stratify_one_by_other_valida_columnas(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    with pytest.raises(AssertionError):
        PairsStatisticalData.stratify_one_by_other(pdi.df, "no_col", "paragraph")

    with pytest.raises(AssertionError):
        PairsStatisticalData.stratify_one_by_other(pdi.df, "sindex", "no_col")
