from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.dataset_interface.helpers import (
    prev_page,
    get_split_separate_laloma_and_letters,
)
from cropgen.shared.default_parameters import max_context_chars
import pandas as pd


def _is_letter(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, str):
        return x.isdigit()
    else:
        raise TypeError(
            f'Se ha detectado un elemento de tipo {type(x)} en la columna "page" del dataframe.'
        )


class DatasetInterface:
    def __init__(self, paths: PathBundle):
        self.df = pd.read_json(paths.json_filepath, lines=True)

        if "is_letter" not in self.df.columns:
            self.df["is_letter"] = self.df.page.apply(
                lambda x: isinstance(x, int) or str(x).isdigit()
            )

        self.df["has_enough_context"] = self.df.apply(
            self._has_enough_context,
            axis=1,
        )
        # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1

        # df_full = self.df[self.df.order == "full"]
        #
        # self.page2fulltext: dict[str, str] = {
        #     page: df_full[df_full.page == page].text.iloc[0]
        #     for page in df_full.page.unique()
        # }

    @property
    def clean_pages(self) -> pd.DataFrame:
        return self.df[
            ~self.df["text"].apply(
                lambda x: not isinstance(x, str) or (x.strip() == "")
            )
        ]

    @property
    def is_clean(self):
        return not any(
            self.df["text"].apply(lambda x: not isinstance(x, str) or (x.strip() == ""))
        )

    def prev_page(self, page: str | int):
        return prev_page(str(page), self.clean_pages)

    def split(
        self, p: float, order_to_consider: tuple[int] = (1,)
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Genera el split train/test, usando un algoritmo greedy sobre cada subconjunto con
        igual proporción deseada: primero LoLoMa y luego las cartas, para asegurar homogeneidad
        en ese respecto."""
        return get_split_separate_laloma_and_letters(self.df, p, order_to_consider)

    def _has_enough_context(self, row):
        return lambda row: not (
            (row.sindex <= max_context_chars) and (self.prev_page(row.page) == False)
        )
