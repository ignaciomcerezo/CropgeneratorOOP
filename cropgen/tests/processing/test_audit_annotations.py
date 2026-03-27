from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.processing.ImageBox import ImageBox
from cropgen.processing.TextFragment import TextFragment
from tqdm.auto import tqdm
from cropgen.processing.Paragraph import Paragraph
from PIL import Image


def _box_checks(box: ImageBox, paragraph: Paragraph | int, ann: AnnotatedPage):
    assert isinstance(box, ImageBox)
    assert isinstance(box.fragment, TextFragment)
    assert isinstance(box.crop, Image.Image)
    assert box.task_id == ann.task_id

    if paragraph != -1:
        assert box.fragment.id in paragraph.text_fragments_ids


def _fragment_checks(
    fragment: TextFragment, paragraph: Paragraph | int, ann: AnnotatedPage
):
    assert isinstance(fragment, TextFragment)
    assert isinstance(fragment.box, ImageBox)
    assert isinstance(fragment.text, str)
    assert fragment.text  # no vacío
    assert isinstance(fragment.text_outside_math(), str)
    assert isinstance(fragment.text_inside_math(), str)
    assert fragment.task_id == ann.task_id

    if paragraph != -1:
        assert fragment.box.id in paragraph.image_boxes_ids


def test_audit_annotations(paths, ls_url, ls_token, lsi):

    for task in tqdm(lsi.simplified_tasks):
        image_path = paths.get_image_path_from_task(task)
        image = Image.open(image_path)
        for ann in task["annotations"]:
            ann = AnnotatedPage(ann, image, usernames_LS=lsi.usernames)
            ann.assert_pairing()  # esto ya se llama dentro del AnnotatedPage.__init__(), pero por asegurar

            seen_boxes = set()
            seen_fragments = set()

            for paragraph in ann.paragraphs:
                seen_boxes_par = set()  # TODO check equals
                seen_fragments_par = set()

                assert isinstance(paragraph, Paragraph)
                assert len(paragraph.image_boxes_ids) != 0
                assert len(paragraph.image_boxes_ids) == len(
                    paragraph.text_fragments_ids
                )
                assert len(paragraph.image_boxes) == len(paragraph.text_fragments)
                assert len(paragraph.image_boxes_ids) == len(paragraph.image_boxes)

                for image_box in paragraph.image_boxes:
                    seen_boxes_par.add(image_box.id)
                    _box_checks(image_box, paragraph, ann)

                for text_fragment in paragraph.text_fragments:
                    seen_fragments_par.add(text_fragment.id)
                    _fragment_checks(text_fragment, paragraph, ann)

                assert seen_boxes_par == set(paragraph.image_boxes_ids)
                assert seen_fragments_par == set(paragraph.text_fragments_ids)

                set_keys = set(paragraph.image_boxes_ids)

                assert set_keys == seen_boxes_par

                for key in paragraph.subgraph:
                    assert paragraph.subgraph[key].issubset(set_keys)

                seen_boxes = seen_boxes.union(seen_boxes_par)
                seen_fragments = seen_fragments.union(seen_fragments_par)

            for text_fragment in ann.fragments_without_paragraph():

                assert text_fragment.id not in seen_fragments
                seen_fragments.add(text_fragment.id)
                _fragment_checks(text_fragment, -1, ann)

                image_box = text_fragment.box
                assert image_box.id not in seen_boxes
                seen_boxes.add(image_box.id)
                _box_checks(image_box, -1, ann)

            assert seen_boxes == set(ann.image_boxes.keys())
            assert seen_fragments == set(ann.text_fragments.keys())

    assert AnnotatedPage.n_annotation_errors == 0
