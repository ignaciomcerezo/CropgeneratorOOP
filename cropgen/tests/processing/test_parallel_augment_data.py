from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel


def test_augment_data_parallel_orders123(paths, lsi, five_letter_task_numbers):

    paths.clean_output_folder()

    augment_data_parallel(
        paths,
        orders_to_consider=[1, 2, 3],
        generate_full_pages=True,
        generate_paragraphs=True,
        tasks_only=[280, 690],
        lsi=lsi,
        num_processes=2,
    )


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
