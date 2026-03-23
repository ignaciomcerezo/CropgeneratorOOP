from external_interfaces.LabelStudioInterface import LabelStudioInterface
from processing.AnnotatedPage import AnnotatedPage
from kaggle_integration.PathBundle import PathBundle
from tqdm.auto import tqdm
from PIL import Image


def test_paragraph_subgraphs():
    paths = PathBundle()
    lsi = LabelStudioInterface(paths)

    for task in tqdm(lsi.simplified_tasks[:50] + lsi.simplified_tasks[-50:]):
        img_path = paths.get_image_path_from_task(task)
        img = Image.open(img_path)
        for Ann in (
            AnnotatedPage(ann, img, unrotate=False, usernames_LS=lsi.usernames)
            for ann in task["annotations"]
        ):

            for paragraph in Ann.paragraphs:

                set_keys = set(paragraph.image_boxes_ids)

                for key in paragraph.subgraph:
                    assert paragraph.subgraph[key].issubset(set_keys)
