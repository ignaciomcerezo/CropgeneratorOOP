from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
import os

# def test_parallel_augment_data(paths, lsi, task_macedonia):
#     augment_data_parallel(paths, [1], True, True, tasks_only=task_macedonia, lsi=lsi)
#


def test_augment_data_parallel(paths, lsi, task_macedonia):

    paths.clean_output_folder()

    augment_data_parallel(
        paths,
        orders_to_consider=[1],
        generate_full_pages=True,
        generate_paragraphs=True,
        tasks_only=task_macedonia,
        lsi=lsi,
        num_processes=2,
    )
