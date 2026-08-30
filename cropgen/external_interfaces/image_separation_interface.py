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

    def parts_required(self):
        return ["raw_images"]

    def parts_managed(self):
        return ["background_images", "stroke_images"]

    def setup(self):
        for raw_image_path in tqdm(self.paths.raw_images_path.iterdir()):

            raw_image = Image.open(raw_image_path)

            background, stroke = separate_background_and_stroke(
                raw_image,
                out_longest_side=DATASET_LONGEST_SIZE_PX,
                processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
            )
            stroke.save(
                self.paths.stroke_images_path
                / f"{raw_image_path.stem}{raw_image_path.suffix}"
            )
            background.save(
                self.paths.background_images_path
                / f"{raw_image_path.stem}{raw_image_path.suffix}"
            )
