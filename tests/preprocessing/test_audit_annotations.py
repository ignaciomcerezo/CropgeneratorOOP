from preprocessing.AnnotatedPage import AnnotatedPage
from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.helpers.PairingErrors import PairingError
from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from PIL import Image
from tqdm.auto import tqdm


def test_audit_annotations():
    LSI = LabelStudioInterface()

    for task in tqdm(LSI.simplified_tasks):
        image_path = get_image_path_from_task(task)
        image = Image.open(image_path)
        for ann in task["annotations"]:
            Ann = AnnotatedPage(ann, image)
            Ann.assert_pairing()  # esto ya se llama dentro del AnnotatedPage.__init__(), pero por asegurar

    assert AnnotatedPage.n_annotation_errors == 0
