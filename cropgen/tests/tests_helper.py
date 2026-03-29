from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
from PIL import Image


def load_particular_annotation(
    paths: PathBundle,
    task_id: int,
    annotation_number_in_task: int = 0,
    lsi: LabelStudioInterface | None = None,
    reload_lsi=False,
) -> AnnotatedPage:
    """
    Carga la anotación annotation_number_in_task-ésima de la tarea task_id, y la devuelve como una instancia
    de la clase AnnotatedPage
    """

    if reload_lsi:
        LabelStudioInterface.update_conditional(paths)
        lsi = LabelStudioInterface(paths)
    else:

        lsi = lsi if lsi else LabelStudioInterface(paths)

    tsk = lsi[task_id][annotation_number_in_task]

    task = [task for task in lsi.raw_tasks if task.get("id") == int(task_id)][0]
    img_path = paths.get_image_path_from_task(task)
    # img_path = images_path / f"{str(task_id).rjust(3,"0")}.png"
    ann = AnnotatedPage(
        tsk,
        Image.open(img_path),
        False,
        usernames_labelstudio=lsi.usernames,
    )
    return ann
