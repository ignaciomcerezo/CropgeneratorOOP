import cv2
from cropgen.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from cropgen.external_interfaces.external_interface import ExternalInterface
from cropgen.shared.path_bundle import PathBundle
from cropgen.shared.image_processing import separate_background_and_stroke
from tqdm.auto import tqdm
import numpy as np


class ImageSeparationInterface(ExternalInterface):
    def __init__(self):
        pass

    def __repr__(self):
        return f"<ImageSeparationInterface'>."

    def parts_required(self):
        return {"raw_images"}

    def parts_managed(self):
        return {"background_images", "stroke_images"}

    def setup(self, paths: PathBundle):
        for raw_image_path in tqdm(
            list(paths.raw_images_path.iterdir()),
            desc="Stroke/background separation...",
        ):

            raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

            if raw_image is None:
                raise ValueError("Tried to separate nonexistent image.")

            if paths.has_processed_images(raw_image_path.stem):
                continue

            background, stroke = separate_background_and_stroke(
                raw_image,
                out_longest_side=DATASET_LONGEST_SIZE_PX,
                processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
            )
            cv2.imwrite(
                PathBundle.change_image_category_path(raw_image_path, "stroke"), stroke
            )
            cv2.imwrite(
                PathBundle.change_image_category_path(raw_image_path, "background"),
                background,
            )
