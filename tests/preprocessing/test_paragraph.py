from tests.tests_helper import load_particular_annotation
from labelstudio.LabelStudioInterface import LabelStudioInterface

n_paragraph_tasks: list[list[int | tuple]] = []
n_paragraph_tasks.append([1, 2, 3, 4, 5, 6, 7, 8, (11, 0), 13, 14, 16, 17, 18, (11, 1)])
n_paragraph_tasks.append([9, 10, 12, 15])

lsi = LabelStudioInterface()

for n_minus_one, task_group in enumerate(n_paragraph_tasks):
    for i, element in enumerate(task_group):
        if isinstance(element, int):
            n_par = load_particular_annotation(element, 0, lsi=lsi).n_paragraphs
            assert n_par == (
                n_minus_one + 1
            ), f"Se esperaban {n_minus_one +1} párrafos en la anotación {(element, 0)}, pero tiene {n_par}."
        else:
            n_par = load_particular_annotation(*element, lsi=lsi).n_paragraphs
            assert n_par == (
                n_minus_one + 1
            ), f"Se esperaban {n_minus_one +1} párrafos en la anotación {element}, pero tiene {n_par}."
