from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from pytest import skip
import pandas as pd


def test_audit_splitter_loader(paths: PathBundle):
    if not paths.json_filepath.exists():
        skip(
            "Para poder probar si se cargan correctamente los crops y sus transcripciones debe haber alguno."
        )
    else:
        df = pd.read_json(paths.json_filepath, lines=True)
        df_notvalid = df[
            df["text"].apply(lambda x: not isinstance(x, str) or (x.strip() == ""))
        ]

        assert len(df_notvalid) == 0

        # si lo anterior ha pasado, lo siguiente debería también:
        pdi = PairsDataInterface(paths)
        assert pdi.is_clean
