from debugpy.launcher.debuggee import process
from PIL import Image

from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.processing import AnnotatedPage
from cropgen.external_interfaces.label_studio.ls_typed_dicts import (
    LabelStudioTask,
    ImageBaseResult,
    SimplifiedTask,
)
from cropgen.shared.path_bundle import PathBundle
from cropgen.shared.image_processing import separate_background_and_stroke
from cropgen.tests.object_mothers import mother_pil_image


def task_from_task_id(lsi: LabelStudioInterface, task_id: int | str) -> LabelStudioTask:
    return [task for task in lsi.raw_tasks if task.id == int(task_id)][0]


def load_ann(
    paths: PathBundle,
    task_id: int,
    annotation_number_in_task: int = 0,
) -> AnnotatedPage:
    """
    Carga la anotación annotation_number_in_task-ésima de la tarea task_id, y la devuelve como una instancia
    de la clase AnnotatedPage
    """

    anns = AnnotatedPage.from_path_bundle(
        paths, tasks=[task_id], combine_same_page_annotations=False
    )

    if len(anns) <= annotation_number_in_task:
        raise ValueError(
            "Asked for an annotation's index greater than the number"
            f" of annotations in task {task_id}, {len(anns)}<{annotation_number_in_task}."
        )

    return anns[annotation_number_in_task]


def extract_height_width_from_task(
    task: SimplifiedTask | LabelStudioTask,
) -> tuple[int, int]:
    """
    Devuelve width, height en ese orden
    """
    result = None

    for result in task.annotations[0].result:
        if issubclass(result.__class__, ImageBaseResult) or isinstance(
            result, ImageBaseResult
        ):
            break

    if not isinstance(result, ImageBaseResult):
        raise ValueError(f"No image boxes in  (Task {task}).")

    retrieved_width = result.original_width
    retrieved_height = result.original_height
    return retrieved_width, retrieved_height
