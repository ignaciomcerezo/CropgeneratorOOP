from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.AnnotatedPage import AnnotatedPage
from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from PIL import Image


def load_particular_annotation(
    task_id: int,
    annotation_number_in_task: int = 0,
    lsi: LabelStudioInterface | None = None,
) -> AnnotatedPage:
    """
    Carga la anotación annotation_number_in_task-ésima de la tarea task_id, y la devuelve como una instancia
    de la clase AnnotatedPage
    """

    lsi = lsi if lsi else LabelStudioInterface()
    tsk = lsi[task_id][annotation_number_in_task]

    task = [task for task in lsi.raw_tasks if task.get("id") == int(task_id)][0]
    img_path = get_image_path_from_task(task)
    # img_path = images_path / f"{str(task_id).rjust(3,"0")}.png"
    ann = AnnotatedPage(
        tsk,
        Image.open(img_path),
        False,
    )
    return ann
