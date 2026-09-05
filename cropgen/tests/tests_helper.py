from cropgen.ocr_units import OCRPage

from cropgen.shared.path_bundle import PathBundle


def load_ann(
    paths: PathBundle,
    task_id: int,
    annotation_number_in_task: int = 0,
) -> OCRPage:
    """
    Carga la anotación annotation_number_in_task-ésima de la tarea task_id, y la devuelve como una instancia
    de la clase AnnotatedPage
    """

    anns = OCRPage.from_path_bundle(
        paths, tasks=[task_id], combine_same_page_annotations=False
    )

    if len(anns) <= annotation_number_in_task:
        raise ValueError(
            "Asked for an annotation's index greater than the number"
            f" of annotations in task {task_id}, {len(anns)}<{annotation_number_in_task}."
        )

    return anns[annotation_number_in_task]
