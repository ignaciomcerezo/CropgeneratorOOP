import numpy as np
from fuzzywuzzy import fuzz
import pytest

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.tests.tests_helper import load_particular_annotation


def test_contextualize_by_words(task_macedonia, pdi):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    scores = []

    for task_n in task_macedonia:
        df = pdi.df[pdi.df.task == task_n]
        for _, row in df.iterrows():
            if row.sindex == -1:
                continue

            context = pdi.get_rows_context_by_words(row)
            contextualized = " ".join([context, row.text])

            curr_trans_text = pdi.annid2fulltext[row.id]
            prev_page_n = pdi.prev_page(row.page)

            prev_trans_text = pdi.page2somefulltext[
                prev_page_n
            ]  # si no hay, devuelve False, que p2sft -> ""

            reference = " ".join([prev_trans_text, curr_trans_text])

            score = fuzz.partial_ratio(contextualized, reference)
            assert score >= 95, (
                f"Detectado bajo {score=}:\n\t {row.task=}, {row.id=}, {row.sindex}"
                f"contextualized =\n{contextualized}\n\n"
                f"reference=\n{reference}\n\n"
            )

    if scores:
        print(f"Min score: {np.min(scores):.2f}")
        print(f"Avg score: {np.mean(scores):.2f}")
        print(f"Max score: {np.max(scores):.2f}")
    else:
        print("No scores computed.")


def _get_annotation(task_id: int, lsi: LabelStudioInterface):
    tasks_matching = [task for task in lsi.simplified_tasks if task.id == task_id]
    assert len(tasks_matching) != 1, f"{len(tasks_matching)=}"
    return tasks_matching[0]


def test_cluster_in_contextualize(
    paths, task_macedonia: list[int], pdi: PairsDataInterface, lsi: LabelStudioInterface
):
    if pdi is None:
        pytest.skip("pairs.jsonl no existe")

    for task_n in task_macedonia:
        df = pdi.df[pdi.df.task == task_n]

        if df.empty:
            raise ValueError(f"No existe la tarea {task_n} en el dataframe")

        annotations = [
            load_particular_annotation(paths, task_n, k, lsi=lsi)
            for k in range(len(lsi[task_n]))
        ]
        anns_by_id = {ann.annotation_unique_id: ann for ann in annotations}

        for ann_id, df_ann in df.groupby("id"):
            ann = anns_by_id.get(int(ann_id))
            if ann is None:
                continue

            order_1_rows = df_ann[(df_ann.order == 1) & (df_ann.sindex != 0)]

            for _, row in order_1_rows.iterrows():
                fragments = [
                    fragment
                    for fragment in ann.text_fragments.values()
                    if fragment.starting_index == int(row.sindex)
                ]

                if not fragments:
                    continue

                if len(fragments) > 1:
                    text_matches = [f for f in fragments if f.text == row.text]
                    fragment = text_matches[0] if text_matches else fragments[0]
                else:
                    fragment = fragments[0]

                _, clustered_text, _ = ann.cluster_reading_order([fragment.box.id])

                context = pdi.get_rows_context_by_words(row)
                longer_text = " ".join(
                    [piece for piece in [context, row.text] if isinstance(piece, str)]
                ).strip()

                score = fuzz.partial_ratio(clustered_text, longer_text)
                if score < 90:
                    print(
                        f"Low score={score} | task={task_n} ann_id={ann_id} sindex={row.sindex}"
                        f"\nclustered={clustered_text}"
                        f"\nreference={longer_text}\n"
                    )
