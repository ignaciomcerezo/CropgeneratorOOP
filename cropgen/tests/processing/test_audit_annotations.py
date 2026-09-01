from cropgen.shared.path_bundle import PathBundle
from cropgen.processing.helpers.helper_to_classes import is_path_graph
from debugpy.launcher.debuggee import process
from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
import re

import pytest
from PIL import Image
from shapely import Polygon, MultiPolygon
from tqdm.auto import tqdm

from cropgen.processing import AnnotatedPage, Line, Paragraph
from cropgen.tests.object_mothers import mother_pil_image
from cropgen.tests.tests_helper import extract_height_width_from_task


def _line_checks(line: Line, paragraph: Paragraph | int, ann: AnnotatedPage):
    errmsg = f"Error with line {line} and ann {ann}"
    assert isinstance(line, Line), errmsg
    assert isinstance(line.stroke_crop, Image.Image), errmsg
    assert isinstance(line.task_id, int), errmsg
    assert isinstance(
        line.polygon, (Polygon, MultiPolygon)
    ), errmsg  # que pueda ser un multipolygon es una consecuencia de usar el módulo
    assert isinstance(line.index, int), errmsg

    assert line.task_id == ann.task_id, errmsg
    if paragraph != -1:
        assert isinstance(paragraph, Paragraph), errmsg
        assert line.id in paragraph.line_ids, errmsg

    assert isinstance(line.text, str), errmsg
    assert line.text.strip(), errmsg  # no vacío
    assert line.task_id == ann.task_id, errmsg
    assert isinstance(line.starting_index, int), errmsg


def _compose_error_msg_sindices(ann: AnnotatedPage) -> str:
    msg = f"""
    No todos framgentos tienen asociado un int como starting_index: 
    {[x.starting_index for x in ann.lines.values()]}. Son los siguientes:"""

    for line in ann.lines.values():
        if line.starting_index is None:
            msg += "\n\t > " + line.text

    return msg


def lines_without_paragraph(annotated_page: AnnotatedPage) -> list[Line]:
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

    for ann in tqdm(AnnotatedPage.from_path_bundle(paths)):
        seen_lines = set()
        first_sindices_of_paragraphs = []

        sindices = [x.starting_index for x in ann.lines.values()]
        if not all(isinstance(x, int) for x in sindices):
            raise AssertionError(_compose_error_msg_sindices(ann))

        ann_graph_keys = set(ann.graph.keys())

        for paragraph in ann.paragraphs:
            errmsg = f"Error with ann {ann} in paragraph {paragraph}"
            assert isinstance(paragraph, Paragraph), errmsg

            assert paragraph.subgraph is not None, errmsg
            assert is_path_graph(paragraph.subgraph), errmsg

            for order in range(len(paragraph)):
                for subsubgraph_keys in paragraph.generate_connected_subgraphs(order):
                    assert set(subsubgraph_keys).issubset(ann_graph_keys), errmsg

            seen_lines_par = set()

            assert len(paragraph.line_ids) != 0, errmsg
            assert len(paragraph.line_ids) == len(paragraph.lines), errmsg

            for line in paragraph.lines:
                seen_lines_par.add(line.id)
                _line_checks(line, paragraph, ann)

            assert seen_lines_par == set(paragraph.line_ids), errmsg

            set_keys = set(paragraph.line_ids)
            assert isinstance(paragraph.subgraph, dict), errmsg

            for key in paragraph.subgraph.keys():
                assert paragraph.subgraph[key].issubset(set_keys), errmsg

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

    assert AnnotatedPage.n_annotation_errors == 0


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
