from cropgen.processing.annotated_page import AnnotatedPage
from cropgen.processing.line import Line
from typing import Sequence
import pytest
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
from cropgen.tests.tests_helper import load_ann

one_paragraph = [1, 2, 3, 4, (11, 0), 13, 14, 17, 18, (11, 1)]
two_paragraphs = [9, 10, 12, 15, 16]

format = lambda x: x if isinstance(x, tuple) else (x, 0)


n_paragraphs = [1] * len(one_paragraph) + [2] * len(two_paragraphs)
pages = [format(x) for x in one_paragraph] + [format(x) for x in two_paragraphs]


@pytest.mark.parametrize(("page", "supposed_paragraphs"), zip(pages, n_paragraphs))
def test_paragraph_v1(
    paths: PathBundle,
    page: tuple[int, int],
    supposed_paragraphs,
):
    lsi: LabelStudioInterface = paths.lsi  # ty: ignore[invalid-assignment]
    ann: AnnotatedPage = lsi.get_annotated_page(*page)
    n_par = len(ann.paragraphs)
    assert n_par == (
        supposed_paragraphs
    ), f"Se esperaban {supposed_paragraphs} párrafos en la anotación {ann} de la página {page}, pero tiene {n_par}."


def test_paragraph_v2(
    paths,
    five_letter_task_numbers,
    five_laloma_task_numbers,
    two_paragraph_laloma,
    three_paragraph_laloma,
):

    for task_n in five_letter_task_numbers + five_laloma_task_numbers:
        print(f"1 // Checking {task_n=}")
        ann = load_ann(paths, task_n, fake_image=True)
        assert len(ann.paragraphs) == 1

    for task_n in two_paragraph_laloma:
        print(f"2 // Checking {task_n=}")
        ann = load_ann(paths, task_n, fake_image=True)
        assert len(ann.paragraphs) == 2

    for task_n in three_paragraph_laloma:
        print(f"3 // Checking {task_n=}")
        ann = load_ann(paths, task_n, fake_image=True)
        assert len(ann.paragraphs) == 3


def test_paragraph_ordering_v3(paths):
    ann = load_ann(paths, 280, fake_image=True)
    assert len(ann.paragraphs) == 1

    assert [line.box_id for line in ann.paragraphs[0]] == [
        "3vLJQ-OQfx",
        "0mE8YfO-qb",
        "fI2od0TJYp",
        "-i8tVxsXKk",
        "2Kg1W6xu_o",
        "EqWM_bDMj3",
        "MlxoXDWETl",
        "MENHQhxGbX",
        "G7ISJaC5B3",
        "7r2YhWQDTz",
        "mCWcKtnWy7",
        "FgD1VOifzq",
        "TcqI79fmwV",
    ]

    ann = load_ann(paths, 690, fake_image=True)

    assert len(ann.paragraphs) == 2

    assert [line.box_id for line in ann.paragraphs[0]][:6] == [
        "XhDbxw40iQ",
        "gIMBZ5nlKa",
        "HkIgHqMJAY",
        "-Nu-LoVU7L",
        "x1z_rZLPBn",
        "jMeItlFlAT",
    ]

    assert [line.box_id for line in ann.paragraphs[1]] == [
        "fdroAOvxV0",
        "9iaENiPLJf",
        "RYvD2P4Yso",
        "TbGcgLEoHR",
        "q5aAvQGbt_",
    ]

    ann = load_ann(paths, 332, annotation_number_in_task=1, fake_image=True)

    assert [line.box_id for line in ann.paragraphs[0]] == [
        "KotWxsgS87",
        "ydPjZ4UtEq",
        "9mI-CXY7JI",
        "sES-y04ZNp",
        "8orwKeStWy",
        "1drw-LKT-G",
        "EjbYa0Bqrz",
        "HGvNRU0DCD",
        "LehE7XMpFp",
        "dNal3edaPF",
    ]
