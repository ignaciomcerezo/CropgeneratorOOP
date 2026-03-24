from cropgen.tests.tests_helper import load_particular_annotation
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface


def test_paragraph(paths):
    n_paragraph_tasks: list[list[int | tuple]] = [
        [1, 2, 3, 4, 5, 6, 7, 8, (11, 0), 13, 14, 16, 17, 18, (11, 1)],
        [9, 10, 12, 15],
    ]

    lsi = LabelStudioInterface(paths)

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
