from processing.AnnotatedPage import AnnotatedPage
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from shared.PathBundle import PathBundle
from PIL import Image


def test_audit_annotations(paths, ls_url, ls_token, lsi):

    for task in lsi.simplified_tasks[:50] + lsi.simplified_tasks[-50:]:
        image_path = paths.get_image_path_from_task(task)
        image = Image.open(image_path)
        for ann in task["annotations"]:
            ann = AnnotatedPage(ann, image, usernames_LS=lsi.usernames)
            ann.assert_pairing()  # esto ya se llama dentro del AnnotatedPage.__init__(), pero por asegurar

    assert AnnotatedPage.n_annotation_errors == 0
