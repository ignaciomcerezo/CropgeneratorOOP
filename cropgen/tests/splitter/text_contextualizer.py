def test_contextualize_by_words(task_macedonia, pdi):
    for task_n in task_macedonia:
        df = pdi.df[pdi.df.task == task_n]
