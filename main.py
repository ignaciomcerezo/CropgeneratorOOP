from processing.augment_data.augment_data_sequential import augment_data_sequential
from paths import simplified_filepath, raw_export_filepath

from preprocessing.simplify.simplify_export import simplify_export

from downloaders.LabelStudioInterface import LabelStudioInterface

LSinterface = LabelStudioInterface(export_path=raw_export_filepath)
LSinterface.save_export()

simplify_export(simplified_filepath= simplified_filepath, raw_export_filepath= raw_export_filepath)


augment_data_sequential(simplified_filepath, orders_to_consider=[1,2,3], task_only = [5,6,7])


