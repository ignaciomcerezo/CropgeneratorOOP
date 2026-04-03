import pandas as pd
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pytest


def test_pdi_clean_pages_and_is_clean(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    clean = pdi.clean_pages
    assert isinstance(clean, pd.DataFrame)
    assert pdi.is_clean is True
    # clean_pages debe ser subconjunto de df
    assert set(clean.index).issubset(set(pdi.df.index))


def test_pdi_prev_page(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    # Prueba con páginas numéricas y de cartas
    pages = list(pdi._pages)
    for page in pages[:5]:
        prev = pdi.prev_page(str(page))
        assert prev is False or isinstance(prev, str)


def test_pdi_split(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    train, test = pdi.split(0.2)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(pdi.df)


def test_pdi_get_rows_context_by_words_and_chars(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    row = pdi.df.iloc[0]
    ctx_words = pdi.get_rows_context_by_words(row)
    ctx_chars = pdi.get_rows_context_by_chars(row)
    assert isinstance(ctx_words, str)
    assert isinstance(ctx_chars, str)


def test_pdi_build_mappings_and_properties(pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    # Verifica que los diccionarios de mapeo están bien formados
    assert isinstance(pdi.page2somefulltext, dict)
    assert isinstance(pdi.annid2fulltext, dict)
    assert len(pdi.page2somefulltext) > 0
    assert len(pdi.annid2fulltext) > 0
