from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.tests.tests_helper import load_particular_annotation
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface


def test_paragraph_v1(paths, lsi):
    n_paragraph_tasks: list[list[int | tuple]] = [
        [1, 2, 3, 4, 5, 6, 7, 8, (11, 0), 13, 14, 16, 17, 18, (11, 1)],
        [9, 10, 12, 15],
    ]

    for n, task_group in enumerate(n_paragraph_tasks, start=1):
        for i, element in enumerate(task_group):
            if isinstance(element, int):
                n_par = load_particular_annotation(
                    paths, element, 0, lsi=lsi
                ).n_paragraphs
                assert (
                    n_par == n
                ), f"Se esperaban {n} párrafos en la anotación {(element, 0)}, pero tiene {n_par}."
            else:
                n_par = load_particular_annotation(
                    paths, *element, lsi=lsi
                ).n_paragraphs
                assert n_par == (
                    n
                ), f"Se esperaban {n} párrafos en la anotación {element}, pero tiene {n_par}."

    ann30 = load_particular_annotation(paths, 30)

    assert len(ann30.paragraphs) == 2


def test_paragraph_v2(
    paths,
    five_letter_task_numbers,
    five_laloma_task_numbers,
    two_paragraph_laloma,
    three_paragraph_laloma,
    lsi,
):

    for task_n in five_letter_task_numbers + five_laloma_task_numbers:
        print(f"1 // Checking {task_n=}")
        ann = load_particular_annotation(paths, task_n, lsi=lsi)
        assert len(ann.paragraphs) == 1

    for task_n in two_paragraph_laloma:
        print(f"2 // Checking {task_n=}")
        ann = load_particular_annotation(paths, task_n, lsi=lsi)
        assert len(ann.paragraphs) == 2

    for task_n in three_paragraph_laloma:
        print(f"3 // Checking {task_n=}")
        ann = load_particular_annotation(paths, task_n, lsi=lsi)
        assert len(ann.paragraphs) == 3
