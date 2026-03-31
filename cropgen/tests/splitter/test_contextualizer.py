import numpy as np
from fuzzywuzzy import fuzz


def test_contextualize_by_words(task_macedonia, pdi):
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
