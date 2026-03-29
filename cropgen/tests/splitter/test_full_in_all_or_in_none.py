from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pandas as pd


def test_full_in_all_or_in_none(paths):
    dsi = PairsDataInterface(paths)

    if "full" in dsi.df.order:

        for page in pd.unique(dsi.df.page):
            assert "full" in pd.unique(dsi.df.order[dsi.df.page == page])
        print('All have "full"')
    else:
        print('None have "full".')
