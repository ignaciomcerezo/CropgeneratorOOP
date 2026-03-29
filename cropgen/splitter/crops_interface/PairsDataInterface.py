from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.helpers import (
    get_split_separate_laloma_and_letters,
)
import re
from cropgen.shared.default_parameters import max_context_chars
import pandas as pd
from tqdm.auto import tqdm


def _is_letter(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, str):
        return x.isdigit()
    else:
        raise TypeError(
            f'Se ha detectado un elemento de tipo {type(x)} en la columna "page" del dataframe.'
        )


_re_png = re.compile(r"_p(\d*).png$")
_re_plain = re.compile(r"_p(\d*)$")


class PairsDataInterface:
    """
    Interfaz intermedia entre el archivo pairs y la generación del dataset. A rasgos generales, simplemente simplifica
    algo de la complejidad de trabajar con el archivo original, implementando los métodos necesarios para construir
    el dataset "real" (instancia de datasets.Dataset) en splitter.generation.
    """

    __slots__ = ("df", "page2fulltext", "ids", "_pages")

    def __init__(self, paths: PathBundle):
        self.df = pd.read_json(paths.json_filepath, lines=True)

        clean_pages = self.clean_pages
        self._pages = pd.unique(clean_pages.page)
        self.ids = pd.unique(clean_pages.id)

        if "is_letter" not in self.df.columns:
            self.df["is_letter"] = self.df.page.apply(
                lambda x: isinstance(x, int) or str(x).isdigit()
            )

        self.df["has_enough_context"] = self.df.apply(
            lambda x: self._has_enough_context(x),
            axis=1,
        )
        # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
        self.page2fulltext: dict[int | str, str] = dict()

        full = self.df[self.df.order == "full"]

        for page in self._pages:
            # en caso de que haya varias transcripciones anteriores, escogemos la más larga.
            fulls_this_page = full[full.page == page]

            self.page2fulltext[page] = self._choose_longest_prev_transcription(
                page, fulls_this_page
            )

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

    @staticmethod
    def _choose_longest_prev_transcription(
        page: str, fulls_this_page: pd.DataFrame
    ) -> str:
        texts: list[str] = sorted(list(fulls_this_page.text), key=len)
        if len(texts) == 0:
            raise ValueError(f"No hay transcripciones completas para la página {page}")
        return texts[0]

    def prev_page(self, page: str | int) -> str | bool:

        if page.isdigit():  # si es una página de LaLoMa

            prev = str(int(page) - 1).rjust(3, "0")

            if prev in self._pages:
                return prev
            else:
                return False

        else:  # si no es una página de LaLoMa
            matched = _re_png.search(page) or _re_plain.search(page)

            if matched is None:
                raise ValueError(
                    f"No tiene el formato de página esperado: {page=}, {matched=}"
                )

            subpage = int(
                matched.group(1)
            )  # las cartas tienen estructura nombre_p\d*.png, donde \d* son dígitos.
            if subpage > 1:
                w = page.replace(f"_p{subpage}", f"_p{subpage-1}")

                if w in self._pages:
                    return w
                else:
                    return False
            else:
                return False

    def split(
        self, p: float, order_to_consider: tuple[int] = (1,)
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Genera el split train/test, usando un algoritmo greedy sobre cada subconjunto con
        igual proporción deseada: primero LoLoMa y luego las cartas, para asegurar homogeneidad
        en ese respecto."""
        return get_split_separate_laloma_and_letters(self.df, p, order_to_consider)

    def _has_enough_context(self, row: pd.Series):
        # aquí implementamos que aquello que los trocitos desconectados (star nodes) no tienen contexto posible (sindex = -1)
        return not (
            (row.sindex <= max_context_chars) and (self.prev_page(row.page) == False)
        ) and (row.sindex != -1)

    def contextualize_by_words(
        self, row: pd.Series, n_words: int = None, n_words_min: int = None
    ):
        curr_page_n = row.page
        prev_page_n = self.prev_page(curr_page_n)

        if n_words < n_words_min:
            raise ValueError(f"{n_words=} < {n_words_min=}")
        text_curr_page = self.page2fulltext[curr_page_n][: row.sindex]

        words_curr_page = text_curr_page.split()

        if (len(words_curr_page) >= n_words) or not prev_page_n:
            return " ".join(words_curr_page[-n_words:])

        n_needed_words_prev = n_words - len(words_curr_page)

        text_prev_page = self.page2fulltext[prev_page_n]
        words_prev_page = text_prev_page.split()

        return " ".join(words_prev_page[-n_needed_words_prev:] + words_curr_page)
