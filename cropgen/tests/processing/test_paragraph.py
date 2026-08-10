import pytest
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
from cropgen.tests.tests_helper import load_ann

one_paragraph = [1, 2, 3, 4, (11, 0), 13, 14, 16, 17, 18, (11, 1)]
two_paragraphs = [9, 10, 12, 15]

format = lambda x: x if isinstance(x, tuple) else (x, 0)


n_paragraphs = [1] * len(one_paragraph) + [2] * len(two_paragraphs)
pages = [format(x) for x in one_paragraph] + [format(x) for x in two_paragraphs]


@pytest.mark.parametrize(("page", "supposed_paragraphs"), zip(pages, n_paragraphs))
def test_paragraph_v1(
    paths: PathBundle,
    lsi: LabelStudioInterface,
    page: tuple[int, int],
    supposed_paragraphs,
):
    ann = load_ann(paths, *page, fake_image=True)
    n_par = ann.n_paragraphs
    assert n_par == (
        supposed_paragraphs
    ), f"Se esperaban {n_par} párrafos en la anotación {ann} de la página {page}, pero tiene {n_par}."


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
    assert ann.n_paragraphs == 1

    assert ann.paragraphs[0].image_boxes_ids == [
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

    assert ann.n_paragraphs == 2

    assert ann.paragraphs[0].image_boxes_ids[:6] == [
        "XhDbxw40iQ",
        "gIMBZ5nlKa",
        "HkIgHqMJAY",
        "-Nu-LoVU7L",
        "x1z_rZLPBn",
        "jMeItlFlAT",
    ]

    assert ann.paragraphs[1].image_boxes_ids == [
        "fdroAOvxV0",
        "9iaENiPLJf",
        "RYvD2P4Yso",
        "TbGcgLEoHR",
        "q5aAvQGbt_",
    ]

    ann = load_ann(paths, 332, annotation_number_in_task=1, fake_image=True)

    assert ann.paragraphs[0].image_boxes_ids == [
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
