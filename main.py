from processing.sequential.augment_data_sequential_new import (
    augment_data_sequential,
)
from processing.parallel.augment_data_parallel import augment_data_parallel
from shared.PathBundle import PathBundle
from external_interfaces.LabelStudioInterface import LabelStudioInterface

# from downloaders.download_from_bucket import download_all_images
# download_all_images()

paths = PathBundle()

LSinterface = LabelStudioInterface(paths)
LSinterface.save_simplified_export()


def main():
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
    augment_data_parallel(
        orders_to_consider=[1],
        generate_full_pages=True,
        generate_paragraphs=True,
        tasks_only=None,
    )


if __name__ == "__main__":
    main()
