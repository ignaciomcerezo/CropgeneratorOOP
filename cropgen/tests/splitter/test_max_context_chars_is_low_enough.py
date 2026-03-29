from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.shared.default_parameters import max_context_chars


def test_max_context_chars_is_low_enough(paths):
    pdi = PairsDataInterface(paths)

    df_full = pdi.df[pdi.df.order == "full"]
    lower_bound = min(df_full.text.apply(len))
    print("Longitud mínima: ", lower_bound)
    assert (
        lower_bound >= max_context_chars
    ), "El contexto no puede ser mayor que la longitud mínima de alguna página."
