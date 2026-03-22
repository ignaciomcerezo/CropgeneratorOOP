from processing.sequential.augment_data_sequential_new import (
    augment_data_sequential,
)
from paths import simplified_filepath, raw_export_filepath

from labelstudio.simplify_export import simplify_export

from labelstudio.LabelStudioInterface import LabelStudioInterface

from downloaders.download_from_bucket import download_all_images

# download_all_images()
#
LSinterface = LabelStudioInterface(raw_export_filepath=raw_export_filepath)
LSinterface.save_simplified_export()
#
# simplify_export(
#     simplified_filepath=simplified_filepath, raw_export_filepath=raw_export_filepath
# )

# TODO comprobar que los sindex se calculan bien...
augment_data_sequential(
    simplified_filepath,
    orders_to_consider=[1],
    tasks_only=[5, 6, 7],
)


# TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
