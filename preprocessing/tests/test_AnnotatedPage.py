from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.AnnotatedPage import AnnotatedPage
from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from paths import images_path
from PIL import Image


def load_particular_annotation(
    task_id: int, annotation_number_in_task: int = 0
) -> AnnotatedPage:

    lsi = LabelStudioInterface()
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


def show_paragraph_clusters(annotation: AnnotatedPage):
    for par in annotation.paragraphs:
        col, txt, sindex = par.cluster_reading_order()
        col.show()
        print(f" - - - transcription = {txt}")
        print(f" - - - {sindex=}")
        input()


if __name__ == "__main__":
    Annotated_task_5 = load_particular_annotation(5)
    cimg, complete_transcription, _ = Annotated_task_5.cluster_reading_order(
        Annotated_task_5.graph.keys()
    )

    heu_sindex = lambda text: complete_transcription.index(text)

    for image_box in Annotated_task_5.image_boxes.values():

        cimg, ctrans, cindex = Annotated_task_5.cluster_reading_order([image_box.id])

        if cindex != heu_sindex(ctrans):
            print(f"cindex != hey_sindex @{image_box.id}")
        if cindex != heu_sindex(image_box.fragment.text):
            print(f"cindex != hey_sindex @{image_box.id}")
