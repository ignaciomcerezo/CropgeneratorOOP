from src.cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
import shutil
import os


def test_augment_data_parallel(paths, lsi):
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
    augment_data_parallel(
        paths,
        orders_to_consider=[1],
        generate_full_pages=True,
        generate_paragraphs=True,
        tasks_only=[154, 315, 316, 317],
        lsi=lsi,
    )
    shutil.rmtree(paths.crops_path)
    for file in os.listdir(paths.exports_path):
        os.unlink(paths.exports_path / file)
