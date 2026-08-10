from debugpy.launcher.debuggee import process
from PIL import Image

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.shared.LSTypedDicts.aggregates import LabelStudioTask
from cropgen.shared.LSTypedDicts.results import ImageBaseResult
from cropgen.shared.LSTypedDicts.simplified import SimplifiedTask
from cropgen.shared.PathBundle import PathBundle
from cropgen.tests.object_mothers import mother_pil_image


def task_from_task_id(lsi: LabelStudioInterface, task_id: int | str) -> LabelStudioTask:
    return [task for task in lsi.raw_tasks if task.id == int(task_id)][0]


def load_ann(
    paths: PathBundle,
    task_id: int,
    annotation_number_in_task: int = 0,
    process_images: bool = False,
    fake_image: bool = False,
    unrotate: bool = False,
) -> AnnotatedPage:
    """
    Carga la anotación annotation_number_in_task-ésima de la tarea task_id, y la devuelve como una instancia
    de la clase AnnotatedPage
    """

    lsi = paths.lsi

    if lsi is None:
        raise ValueError("The PathBundle instance passed has no .lsi attribute set.")

    simplified_ls_ann = lsi[task_id][annotation_number_in_task]

    task = task_from_task_id(lsi, task_id)

    if not fake_image:
        img_path = paths.get_image_path_from_task(task)
        assert img_path is not None
        img = Image.open(img_path)
    else:
        width, height = extract_height_width_from_task(task)
        img = mother_pil_image(width=width, height=height, color=(255, 0, 255))

    return AnnotatedPage(
        ann=simplified_ls_ann,
        img=img,
        unrotate=unrotate,
        usernames_labelstudio=lsi.usernames,
        process_images=process_images,
    )


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
        raise ValueError(f"No es ha encontrado ningún resultado en la tarea {task}.")

    retrieved_width = result.original_width
    retrieved_height = result.original_height
    return retrieved_width, retrieved_height
