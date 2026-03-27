from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.processing.ImageBox import ImageBox
from cropgen.processing.TextFragment import TextFragment
from tqdm.auto import tqdm
from cropgen.processing.Paragraph import Paragraph
from PIL import Image
from shapely import Polygon, MultiPolygon


def _box_checks(box: ImageBox, paragraph: Paragraph | int, ann: AnnotatedPage):
    assert isinstance(box, ImageBox)
    assert isinstance(box.fragment, TextFragment)
    assert isinstance(box.crop, Image.Image)
    assert isinstance(box.task_id, int)
    assert isinstance(
        box.polygon, (Polygon, MultiPolygon)
    )  # que pueda ser un multipolygon es una consecuencia de usar el módulo
    assert isinstance(box.index, int)

    assert box.task_id == ann.task_id

    assert len(box.associated_fragments) == 1
    if box.true_rectangle:
        assert len(set(box.polygon.exterior.coords)) == 4

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
    assert len(fragment.associated_boxes) == 1
    assert isinstance(fragment.starting_index, int)
    assert isinstance(fragment.word_count, int)
    assert isinstance(fragment.char_count, int)

    if paragraph != -1:
        assert fragment.box.id in paragraph.image_boxes_ids


def _compose_error_msg_sindices(ann: AnnotatedPage) -> str:
    msg = f"No todos framgentos tienen asociado un int como starting_index: {[x.fragment.starting_index for x in ann.image_boxes.values()]}. Son los siguientes:"

    for fragment in ann.text_fragments.values():
        if fragment.starting_index is None:
            msg += "\n\t > " + fragment.text

    return msg


def test_audit_annotations(paths, ls_url, ls_token, lsi):

    for task in tqdm(lsi.simplified_tasks):
        image = Image.open(paths.get_image_path_from_task(task))

        for k_ann, ann in enumerate(task["annotations"]):
            ann = AnnotatedPage(ann, image, usernames_LS=lsi.usernames)
            ann.assert_pairing()  # esto ya se llama dentro del AnnotatedPage.__init__(), pero por asegurar

            seen_boxes = set()
            seen_fragments = set()

            # del antiguo test_sindex
            first_sindices_of_paragraphs = []
            sindices = [x.fragment.starting_index for x in ann.image_boxes.values()]
            if not all(isinstance(x, int) for x in sindices):
                raise AssertionError(_compose_error_msg_sindices(ann))
            # hasta aquí (por ahora)

            for paragraph in ann.paragraphs:
                seen_boxes_par = set()
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

                # antiguo test_sindices
                first_sindices_of_paragraphs.append(
                    paragraph.text_fragments[0].starting_index
                )
                sindices_par_fragments = [
                    f.starting_index for f in paragraph.text_fragments
                ]
                indices_par_boxes = [b.index for b in paragraph.image_boxes]

                assert (
                    -1 not in sindices_par_fragments
                )  # solamente puede haber un -1 si no está conectado a nada

                assert (
                    sorted(sindices_par_fragments) == sindices_par_fragments
                )  # a lo largo del código se asume que vienen así ordenados.

                assert (
                    sorted(indices_par_boxes) == indices_par_boxes
                )  # llevan el mismo orden que los fragmentos

                # antiguo test_paragraph_transcriptions
                _, transcription_1, sindex_1 = ann.cluster_reading_order(
                    paragraph.image_boxes_ids
                )
                transcription_2 = paragraph.transcription()

                assert transcription_1 == transcription_2

            # vuelta al audit_annotations inicial

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
