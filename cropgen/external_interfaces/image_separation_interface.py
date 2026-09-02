from cropgen.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from cropgen.external_interfaces.external_interface import ExternalInterface
from cropgen.shared.path_bundle import PathBundle
from cropgen.shared.image_processing import separate_background_and_stroke
from tqdm.auto import tqdm
from PIL import Image


class ImageSeparationInterface(ExternalInterface):
    def __init__(self, paths: PathBundle):
        self.paths = paths

    def __repr__(self):
        return f"<ImageSeparationInterface with root '{self.paths.root}'>."

    def parts_required(self):
        return {"raw_images"}

    def parts_managed(self):
        return {"background_images", "stroke_images"}

    def setup(self):
        for raw_image_path in tqdm(
            list(self.paths.raw_images_path.iterdir()),
            desc="Stroke/background separation...",
        ):

            raw_image = Image.open(raw_image_path)

            if self.paths.has_processed_images(raw_image_path.stem):
                continue

            background, stroke = separate_background_and_stroke(
                raw_image,
                out_longest_side=DATASET_LONGEST_SIZE_PX,
                processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
            )
            stroke.save(PathBundle.change_image_category_path(raw_image_path, "stroke"))
            background.save(
                PathBundle.change_image_category_path(raw_image_path, "background")
            )
