from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
from cropgen.tests.tests_helper import load_ann


def test_paragraph_v1(paths: PathBundle, lsi: LabelStudioInterface):
    n_paragraph_tasks: list[list[int | tuple]] = [
        [1, 2, 3, 4, 6, 7, (11, 0), 13, 14, 16, 17, 18, (11, 1)],
        [9, 10, 12, 15],
    ]

    for n, task_group in enumerate(n_paragraph_tasks, start=1):
        for i, element in enumerate(task_group):
            if isinstance(element, int):
                n_par = load_ann(
                    paths, element, 0, lsi=lsi, fake_image=True
                ).n_paragraphs
                assert (
                    n_par == n
                ), f"Se esperaban {n} párrafos en la anotación {(element, 0)}, pero tiene {n_par}."
            else:
                n_par = load_ann(paths, *element, lsi=lsi, fake_image=True).n_paragraphs
                assert n_par == (
                    n
                ), f"Se esperaban {n} párrafos en la anotación {element}, pero tiene {n_par}."

    ann30 = load_ann(paths, 30)

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
        ann = load_ann(paths, task_n, lsi=lsi, fake_image=True)
        assert len(ann.paragraphs) == 1

    for task_n in two_paragraph_laloma:
        print(f"2 // Checking {task_n=}")
        ann = load_ann(paths, task_n, lsi=lsi, fake_image=True)
        assert len(ann.paragraphs) == 2

    for task_n in three_paragraph_laloma:
        print(f"3 // Checking {task_n=}")
        ann = load_ann(paths, task_n, lsi=lsi, fake_image=True)
        assert len(ann.paragraphs) == 3


def test_paragraph_ordering_v3(paths, lsi):
    ann = load_ann(paths, 280, lsi=lsi, fake_image=True)
    assert ann.n_paragraphs == 1
    paragraph = ann.paragraphs[0]
    assert paragraph.image_boxes_ids == [
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

    ann = load_ann(paths, 690, lsi=lsi, fake_image=True)

    assert ann.n_paragraphs == 2

    paragraph = ann.paragraphs[0]

    assert paragraph.image_boxes_ids[:11] == [
        "XhDbxw40iQ",
        "gIMBZ5nlKa",
        "HkIgHqMJAY",
        "fdroAOvxV0",
        "9iaENiPLJf",
        "RYvD2P4Yso",
        "TbGcgLEoHR",
        "q5aAvQGbt_",
        "-Nu-LoVU7L",
        "x1z_rZLPBn",
        "jMeItlFlAT",
    ]
