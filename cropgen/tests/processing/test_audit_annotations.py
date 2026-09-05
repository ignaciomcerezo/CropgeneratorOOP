from cropgen.shared.path_bundle import PathBundle
from cropgen.ocr_units.helpers.helper_to_classes import is_path_graph
import re

import pytest
from shapely import Polygon, MultiPolygon
from tqdm.auto import tqdm
import numpy as np
from cropgen.ocr_units import OCRPage, OCRLine, OCRParagraph
from cropgen.tests.object_mothers import mother_image


def _line_checks(line: OCRLine, paragraph: OCRParagraph | int, ann: OCRPage):
    errmsg = f"Error with line {line} and ann {ann}"
    assert isinstance(line, OCRLine), errmsg
    assert isinstance(line.crop, np.ndarray), errmsg
    assert isinstance(line.task_id, int), errmsg
    assert isinstance(
        line.polygon, (Polygon, MultiPolygon)
    ), errmsg  # que pueda ser un multipolygon es una consecuencia de usar el módulo
    assert isinstance(line.index, int), errmsg

    assert line.task_id == ann.task_id, errmsg
    if paragraph != -1:
        assert isinstance(paragraph, OCRParagraph), errmsg
        assert line.id in paragraph.line_ids, errmsg

    assert isinstance(line.text, str), errmsg
    assert line.text.strip(), errmsg  # no vacío
    assert line.task_id == ann.task_id, errmsg
    assert isinstance(line.starting_index, int), errmsg


def _compose_error_msg_sindices(ann: OCRPage) -> str:
    msg = f"""
    No todos framgentos tienen asociado un int como starting_index: 
    {[x.starting_index for x in ann.lines.values()]}. Son los siguientes:"""

    for line in ann.lines.values():
        if line.starting_index is None:
            msg += "\n\t > " + line.text

    return msg


def lines_without_paragraph(annotated_page: OCRPage) -> list[OCRLine]:
    in_paragraph = []
    out_paragraph = []
    for paragraph in annotated_page.paragraphs:
        in_paragraph += [f.id for f in paragraph.lines]
    if len(in_paragraph) == len(annotated_page.lines):
        return []

    for line in annotated_page.lines.values():
        if line.id not in in_paragraph:
            out_paragraph.append(line.id)
    return [annotated_page.lines[line_id] for line_id in out_paragraph]


@pytest.mark.audit
def test_audit_annotations(paths: PathBundle, patch_image_open):

    for ann in tqdm(OCRPage.from_path_bundle(paths)):
        seen_lines = set()
        first_sindices_of_paragraphs = []

        sindices = [x.starting_index for x in ann.lines.values()]
        if not all(isinstance(x, int) for x in sindices):
            raise AssertionError(_compose_error_msg_sindices(ann))

        for paragraph in ann.paragraphs:
            errmsg = f"Error with ann {ann} in paragraph {paragraph}"
            assert isinstance(paragraph, OCRParagraph), errmsg

            assert all(
                [(line.paragraph_index == paragraph.index) for line in paragraph.lines]
            ), f"{[line.paragraph_index for line in paragraph.lines]}, {paragraph.index}"

            seen_lines_par = set()
            # TODO: get back the .is_path_graph check
            assert len(paragraph.line_ids) != 0, errmsg
            assert len(paragraph.line_ids) == len(paragraph.lines), errmsg

            for line in paragraph.lines:
                seen_lines_par.add(line.id)
                _line_checks(line, paragraph, ann)

            assert seen_lines_par == set(paragraph.line_ids), errmsg

            seen_lines.update(paragraph.line_ids)

            first_sindices_of_paragraphs.append(paragraph.lines[0].starting_index)
            sindices_par = [line.starting_index for line in paragraph.lines]

            assert all(isinstance(s, int) for s in sindices_par), errmsg
            assert -1 not in sindices_par, errmsg
            assert sorted(sindices_par) == sindices_par, errmsg

            transcription_1 = ann.synthetic_transcription(paragraph.line_ids), errmsg
            transcription_2 = paragraph.transcription(ann.line_separator), errmsg
            assert transcription_1 == transcription_2

        for line in lines_without_paragraph(ann):
            assert line.id not in seen_lines, errmsg
            seen_lines.add(line.id)
            _line_checks(line, -1, ann)

        assert seen_lines == set(ann.lines.keys()), errmsg

    assert OCRPage.n_annotation_errors == 0


# re_letternumber = re.compile(r"[a-zA-Z]+\d", re.DOTALL)


# @pytest.mark.skip("Esto realmente no es un test")
# def test_letter_number_yuxtaposition(
#     paths: PathBundle, ls_url, ls_token, lsi: LabelStudioInterface
# ):

#     for task in lsi.simplified_tasks:
#         width, height = extract_height_width_from_task(task)
#         stroke = mother_pil_image(width=width, height=height, color=(255, 0, 0))
#         background = mother_pil_image(width=width, height=height, color=(0, 255, 0))

#         for k_ann, ls_ann in enumerate(task.annotations):
#             ann_page = AnnotatedPage(
#                 ls_ann,
#                 stroke,
#                 background,
#                 completer=lsi._get_completer(ls_ann),
#                 updater=lsi._get_updater(ls_ann),
#             )

#             for paragraph in ann_page.paragraphs:
#                 for text_fragment in paragraph.lines:
#                     for match in re_letternumber.findall(text_fragment.text):
#                         print(
#                             f"({ann_page.task_id:>5}|{ann_page.completer:<25}) {text_fragment.id:<5} MATCH: {match:<15}\t<<{text_fragment.text}>> "
#                         )
