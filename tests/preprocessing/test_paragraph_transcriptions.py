from preprocessing.AnnotatedPage import AnnotatedPage
from labelstudio.LabelStudioInterface import LabelStudioInterface

from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from PIL import Image
from tqdm.auto import tqdm


def test_paragraph_transcriptions():

    lsi = LabelStudioInterface()

    for task in tqdm(lsi.simplified_tasks):
        image_path = get_image_path_from_task(task)
        image = Image.open(image_path)
        for Ann in (AnnotatedPage(ann, image) for ann in task["annotations"]):
            for i, paragraph in enumerate(Ann.paragraphs):
                _, transcription_1, sindex_1 = Ann.cluster_reading_order(
                    paragraph.image_boxes_ids
                )
                transcription_2 = paragraph.transcription()

                assert (
                    transcription_1 == transcription_2
                ), f"Las dos transcripciones del párrafo {i} de la tarea {Ann.task_id} {Ann.completer} no coinciden."
