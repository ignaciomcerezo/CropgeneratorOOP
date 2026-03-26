from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel


def test_parallel_augment_data(paths, lsi):
    augment_data_parallel(paths, [1], True, True, tasks_only=[15], lsi=lsi)
