from tests.tests_helper import load_particular_annotation
from preprocessing.AnnotatedPage import AnnotatedPage


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
