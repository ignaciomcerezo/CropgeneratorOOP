import os
import re
from pathlib import Path
from typing import Iterable
from cropgen.shared.LSTypedDicts.simplified import SimplifiedTask

from dotenv import load_dotenv

from cropgen.processing.helpers.text_regularization import french_latex_characters
from cropgen.tests.tests_helper import extract_height_width_from_task
from cropgen.tests.object_mothers import mother_pil_image

# load_dotenv()

from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.processing import AnnotatedPage
from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.online_bucket_interface import OnlineBucketInterface

# paths = PathBundle(Path(os.getcwd()).parents[2])
# obi = (paths)
# obi.update()
# LabelStudioInterface.fetch_and_simplify(paths)
# lsi = LabelStudioInterface(paths)

cmm_p = r"\\[a-zA-Z]+"
wrd_p = r"\w+"

text_nobracket_p = r"[^{}]"
bracketed_block = r"\{" + text_nobracket_p + r"*\}"
for _ in range(10):
    # creamos un bloque de contenido entre llaves - el 10 es un número arbitrario, pero parece razonable.
    bracketed_block = r"\{(?:" + text_nobracket_p + r"|" + bracketed_block + r")*\}"
scriptable_block = rf"({cmm_p}|\w|{bracketed_block})"

subscript_p = rf"\_{scriptable_block}"
superscript_p = rf"\^{scriptable_block}"
foreign_p = rf"[^{''.join([re.escape(char) for char in french_latex_characters])}]"

# _PATTERNS = (r"[a-zA-Z]\d", r"[a-zA-Z]\w", r"\s\;", r"\s\:", r"\\rightarrow")
_PATTERNS = []


def test_undesirable_matches(
    paths: PathBundle, re_patterns: Iterable[str] = _PATTERNS
) -> None:
    assert sum(number_of_matches(paths, re_patterns)) == 0


def number_of_matches(
    paths: PathBundle,
    re_patterns: Iterable[str] | str,
    show_where: bool = True,
    filters: dict[str, list] = {},
) -> list[int]:
    """
    Devuelve el número de ocurrencias de un patrón de regex concreto en las cajas-imagen selccionadas.
    No se hace con el texto concreto, sino en cada línea de forma individual.
    Si show_where, se muestran además qué lugares.
    """

    if isinstance(re_patterns, str):
        re_patterns: list[str] = [re_patterns]

    re_patterns: list[re.Pattern] = [
        re.compile(re_pattern) for re_pattern in re_patterns
    ]

    filters = {x: list(str(y) for y in filters[x]) for x in filters.keys()}

    tasks = paths.lsi.simplified_tasks  # ty: ignore[unresolved-attribute]

    total_matches = [0] * len(re_patterns)

    for task in tasks:
        width, height = extract_height_width_from_task(task)
        stroke = mother_pil_image(width=width, height=height, color=(255, 255, 255))
        background = mother_pil_image(width=width, height=height, color=(255, 255, 255))

        task: SimplifiedTask

        if "id" in filters and (task.id not in filters["id"]):
            continue

        page = paths._get_image_stem_from_task(task)

        if page is None:
            raise ValueError(f"No page found for task {task.id}")

        if "page" in filters and (page not in filters["page"]):
            continue

        for k, ann in enumerate(task.annotations):

            for pttrn_index, re_pattern in enumerate(re_patterns):

                # Primero con unrotate = True (comprobación de los recortes individuales)
                Ann = AnnotatedPage(
                    ann,
                    stroke,
                    background,
                )

                for line in Ann.lines.values():
                    does_match = re_pattern.search(line.text)
                    if does_match and show_where:

                        print(
                            f"Matches in {Ann} for pattern {_str_trimmed(re_pattern.pattern)}:"
                        )
                        starts = []
                        ends = []
                        b_prev = 0

                        print(f"\t{line.text}")
                        print("\t", end="")

                        matches = list(re_pattern.finditer(line.text))
                        for match in matches:
                            a, b = match.span()
                            print(" " * (a - b_prev) + "^" * (b - a), end="")
                            starts.append(a)
                            ends.append(b)
                            b_prev = b
                        print("\n", end="")
                        total_matches[pttrn_index] += len(matches)

    return total_matches


def _str_trimmed(value: str, k: int = 20):
    if len(value) > k:
        return value[: k // 2 + k % 2] + value[-(k // 2) :]
    return value
