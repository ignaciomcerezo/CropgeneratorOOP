from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.helpers import (
    get_split_separate_laloma_and_letters,
)
import re
from typing import Literal
from cropgen.shared.default_parameters import (
    context_chars as default_context_chars,
    context_words as default_context_words,
    min_context_chars as default_min_context_chars,
    min_context_words as default_min_context_words,
)
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


_re_png = re.compile(r"_p(\d*).png$")
_re_plain = re.compile(r"_p(\d*)$")


class PairsDataInterface:
    """
    Interfaz intermedia entre el archivo pairs y la generación del dataset. A rasgos generales, simplemente simplifica
    algo de la complejidad de trabajar con el archivo original, implementando los métodos necesarios para construir
    el dataset "real" (instancia de datasets.Dataset) en splitter.generation.
    """

    __slots__ = (
        "df",
        "_filepath",
        "paths",
        "page2somefulltext",
        "annid2fulltext",
        "ids",
        "_pages",
        "context_words",
        "min_context_words",
        "context_chars",
        "min_context_chars",
    )

    def __init__(
        self,
        paths: PathBundle,
        context_words: int = default_context_words,
        min_context_words: int = default_min_context_words,
        context_chars: int = default_context_chars,
        min_context_chars: int = default_min_context_chars,
    ):
        self.df = pd.read_json(paths.json_filepath, lines=True)
        self._filepath = paths.json_filepath
        self.context_words = context_words
        self.min_context_words = min_context_words
        self.context_chars = context_chars
        self.min_context_chars = min_context_chars

        self._pages = pd.unique(self.clean_pages.page)
        self.ids = pd.unique(self.clean_pages.id)

        self._build_mappings()
        self.paths = paths

        self.df["is_letter"] = self.df.page.apply(
            lambda x: isinstance(x, int) or str(x).isdigit()
        )

        self.df["context_length_chars"] = self.df.apply(
            lambda x: len(self._maximal_context_chars(x)),
            axis=1,
        )
        self.df["context_length_words"] = self.df.apply(
            lambda x: len(self._maximal_context_chars(x).split()) - 1,
            axis=1,
        )

    def _build_mappings(self) -> None:
        self.page2somefulltext: dict[int | str, str] = dict()
        self.page2somefulltext[False] = ""

        full = self.df[self.df.order == "full"]

        for page in self._pages:
            # en caso de que haya varias transcripciones anteriores, escogemos la más larga.
            fulls_this_page = full[full.page == page]

            self.page2somefulltext[page] = self._choose_longest_prev_transcription(
                page, fulls_this_page
            )

        self.annid2fulltext: dict[int, str] = dict()

        for row_id in self.ids:
            self.annid2fulltext[row_id] = full[full.id == row_id].iloc[0].text

    def __repr__(self):
        return f"<PairsDataInterface enlazando con {self._filepath}>"

    @property
    def clean_pages(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame cuyas páginas son aquellas cuyo texto está limpio.
        """
        return self.df[
            ~self.df["text"].apply(
                lambda x: not isinstance(x, str) or (x.strip() == "")
            )
        ]

    @property
    def is_clean(self) -> bool:
        return not any(
            self.df["text"].apply(lambda x: not isinstance(x, str) or (x.strip() == ""))
        )

    @staticmethod
    def _choose_longest_prev_transcription(
        page: str, fulls_this_page: pd.DataFrame
    ) -> str:
        """
        Busca la página anterior. Si la encuentra (i.e. si existe), devuelve el texto completo. Si hay más de una anotación
        de la página previa, devuelve la que tenga la transcripción más larga.
        """
        # noinspection PyTypeChecker
        texts: list[str] = sorted(list(fulls_this_page.text), key=len)
        if len(texts) == 0:
            raise ValueError(f"No hay transcripciones completas para la página {page}")
        return texts[0]

    def prev_page(self, page: int | str) -> str | bool:
        """
        Devuelve el nombre de la página anterior (ya sea si está en LaLoMa o en las cartas). Si no la encuentra, devuelve
        False.
        """

        if isinstance(page, int) or page.isdigit():  # si es una página de LaLoMa

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
        self, p: float, orders_to_consider: list[int] | tuple[int] = (1,)
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Genera el split train/test, usando un algoritmo greedy sobre cada subconjunto con
        igual proporción deseada: primero LoLoMa y luego las cartas, para asegurar homogeneidad
        en ese respecto."""
        return get_split_separate_laloma_and_letters(self.df, p, orders_to_consider)

    @staticmethod
    def _has_enough_context_words(row: pd.Series, threshold: int | None = None) -> bool:
        # aquí implementamos que aquello que los trocitos desconectados (star nodes) no tienen contexto posible (sindex = -1)
        return row.context_length_words >= threshold

    @staticmethod
    def _has_enough_context_chars(row: pd.Series, threshold: int | None = None) -> bool:
        # aquí implementamos que aquello que los trocitos desconectados (star nodes) no tienen contexto posible (sindex = -1)
        return row.context_length_chars >= threshold

    def _maximal_context_chars(self, row: pd.Series) -> str:
        return (
            self.page2somefulltext[self.prev_page(row.page)]
            + self.annid2fulltext[row.id][: row.sindex]
        )

    def get_rows_context_by_words(
        self,
        row: pd.Series,
        n_words: int | None = None,
        n_words_min: int | None = None,
    ) -> str:
        """
        Devuelve el texto anterior al presente en una fila del DataFrame. Se devuelven n_words como máximo y si es posible,
        si no, se devuelven las palabras que haya hasta un mínimo de n_words_min. Si no hay suficientes, devuelve la str vacía.
        """
        n_words = n_words or self.context_words
        n_words_min = n_words_min or self.min_context_words
        if not self._has_enough_context_words(row, n_words_min):
            return ""
        return " ".join(self._maximal_context_chars(row).split()[-n_words:])

    def get_rows_context_by_chars(
        self,
        row: pd.Series,
        n_chars: int | None = None,
        n_chars_min: int | None = None,
    ) -> str:
        """
        Devuelve el texto anterior al presente en una fila del DataFrame. Se devuelven n_words como máximo y si es posible,
        si no, se devuelven las palabras que haya hasta un mínimo de n_words_min. Si no hay suficientes, devuelve la str vacía.
        """
        n_chars = n_chars or self.context_chars
        n_chars_min = n_chars_min or self.min_context_chars
        if not self._has_enough_context_chars(row, n_chars_min):
            return ""
        return self._maximal_context_chars(row)[-n_chars:]
