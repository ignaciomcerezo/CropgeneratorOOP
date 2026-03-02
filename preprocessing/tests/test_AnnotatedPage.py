from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.classes.AnnotatedPage import AnnotatedPage
from paths import images_path
from PIL import Image
from display import display

LSI = LabelStudioInterface()
tsk_id = 5
img_path = images_path / f"{str(tsk_id).rjust(3,"0")}.png"
tsk = LSI[tsk_id][0]
Annotated_task_5 = AnnotatedPage(
    tsk,
    tsk_id,
    Image.open(img_path),
    True,
)

if __name__ == "__main__":
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
