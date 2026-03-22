from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.AnnotatedPage import AnnotatedPage
from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from tqdm.auto import tqdm
from PIL import Image


def test_paragraph_subgraphs():
    lsi = LabelStudioInterface()

    for task in tqdm(lsi.simplified_tasks):
        img_path = get_image_path_from_task(task)
        img = Image.open(img_path)
        for ann in task["annotations"]:
            Ann = AnnotatedPage(ann, img, unrotate=False)

            for paragraph in Ann.paragraphs:

                set_keys = set(paragraph.image_boxes_ids)

                for key in paragraph.subgraph:
                    assert paragraph.subgraph[key].issubset(set_keys)
