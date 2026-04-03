from cropgen.shared.default_parameters import min_context_chars, min_context_words
import pytest


def test_min_context_chars_is_low_enough(pdi, paths):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    df_full = pdi.df[pdi.df.order == "full"]
    lower_bound = min(df_full.text.apply(len))

    print("Longitud mínima: ", lower_bound)
    assert (
        lower_bound >= min_context_chars
    ), "El contexto no puede ser mayor que la longitud mínima de alguna página."


def test_min_context_words_is_low_enough(pdi, paths):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    df_full = pdi.df[pdi.df.order == "full"]
    lower_bound = min(df_full.text.apply(lambda t: len(t.split())))
    print("Cantidad mínima de palabras: ", lower_bound)
    assert (
        lower_bound >= min_context_words
    ), "El contexto de palabras no puede ser mayor que la cantidad mínima de palabras en alguna página."
