import os
import re
from pathlib import Path

from dotenv import load_dotenv

from cropgen.tests.tests_helper import extract_height_width_from_task
from object_mothers import mother_pil_image

load_dotenv()

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from tqdm.auto import tqdm

paths = PathBundle(Path(os.getcwd()).parents[1])
obi = OracleBucketInterface(paths)
obi.update()
LabelStudioInterface.update_conditional(paths)
lsi = LabelStudioInterface(paths)

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

_PATTERNS = [r"[a-zA-Z]\d", r"[a-zA-Z]\w", r"\s\;", r"\s\:", r"\\rightarrow"]


def test_undesirable_matches(re_patterns: list[str] = _PATTERNS) -> None:
    assert number_of_matches(re_patterns) == 0


def number_of_matches(re_patterns: list[str], show_where: bool = True) -> int:
    """
    Devuelve el número de ocurrencias de un patrón de regex concreto en las cajas-imagen selccionadas.
    No se hace con el texto concreto, sino en cada línea de forma individual.
    Si show_where, se muestran además qué lugares.
    """

    re_patterns: list[re.Pattern] = [
        re.compile(re_pattern) for re_pattern in re_patterns
    ]

    tasks = lsi.simplified_tasks

    AnnotatedPage.min_nodes_for_big_box_removal = 500

    total_matches = 0

    for task in tqdm(tasks):
        width, height = extract_height_width_from_task(task)
        img = mother_pil_image(width=width, height=height, color=(255, 0, 255))

        for k, ann in enumerate(task.annotations):

            for re_pattern in re_patterns:

                # Primero con unrotate = True (comprobación de los recortes individuales)
                Ann = AnnotatedPage(ann, img, usernames_labelstudio=lsi.usernames)

                for fragment in Ann.text_fragments.values():
                    does_match = re_pattern.search(fragment.text)
                    if does_match and show_where:

                        print(
                            f"Matches in {Ann} for pattern {_str_trimmed(re_pattern.pattern)}:"
                        )
                        starts = []
                        ends = []
                        b_prev = 0

                        print(f"\t{fragment.text}")
                        print("\t", end="")

                        matches = list(re_pattern.finditer(fragment.text))
                        for match in matches:
                            a, b = match.span()
                            print(" " * (a - b_prev) + "^" * (b - a), end="")
                            starts.append(a)
                            ends.append(b)
                            b_prev = b
                        print("\n", end="")
                        total_matches += len(matches)

    return total_matches


def _str_trimmed(value: str, k: int = 20):
    if len(value) > k:
        return value[: k // 2 + k % 2] + value[-(k // 2) :]
    return value
