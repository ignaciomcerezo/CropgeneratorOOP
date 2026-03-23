from processing.AnnotatedPage import AnnotatedPage
from external_interfaces.LabelStudioInterface import LabelStudioInterface

from kaggle_integration.PathBundle import PathBundle
from PIL import Image
from tqdm.auto import tqdm


def test_paragraph_transcriptions():

    paths = PathBundle()
    lsi = LabelStudioInterface(paths)

    for task in tqdm(lsi.simplified_tasks):
        image_path = paths.get_image_path_from_task(task)
        image = Image.open(image_path)
        for Ann in (
            AnnotatedPage(ann, image, usernames_LS=lsi.usernames)
            for ann in task["annotations"]
        ):
            for i, paragraph in enumerate(Ann.paragraphs):
                _, transcription_1, sindex_1 = Ann.cluster_reading_order(
                    paragraph.image_boxes_ids
                )
                transcription_2 = paragraph.transcription()

                assert (
                    transcription_1 == transcription_2
                ), f"Las dos transcripciones del párrafo {i} de la tarea {Ann.task_id} {Ann.completer} no coinciden."
