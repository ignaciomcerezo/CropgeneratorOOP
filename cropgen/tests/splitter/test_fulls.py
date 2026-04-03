import pandas as pd
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.shared.PathBundle import PathBundle
import pytest


def test_full_in_all_or_in_none(paths: PathBundle, pdi: PairsDataInterface):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    if "full" in pdi.df.order:
        for page in pd.unique(pdi.df.page):
            assert "full" in pd.unique(pdi.df.order[pdi.df.page == page])
        print('All have "full"')
    else:
        print('None have "full".')


def test_at_most_one_full_per_id(paths, pdi):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    full = pdi.df[pdi.df.order == "full"]

    for ann_id in pdi.ids:
        assert (
            len(full[full.id == ann_id]) <= 1
        ), f"Hay {len(full[full.ids == ann_id])} fulls en {ann_id=}"
