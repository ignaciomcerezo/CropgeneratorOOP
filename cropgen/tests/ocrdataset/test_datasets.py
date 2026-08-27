from shapely.geometry import Polygon
from cropgen.ocrdataset.segmentation_dataset import SegmentationDataset
from cropgen.transforms.on_the_fly_transform_pack import OCROnTheFlyTransformPack
from cropgen.transforms.intraparagraph_transforms.horizontal_movement import (
    HorizontalMovement,
)
from cropgen.transforms.intraparagraph_transforms.paragraph_linewise_rotation import (
    ParagraphLinewiseRotation,
)
from cropgen.shared.parameters import UniformDistribution
from cropgen.transforms.intraparagraph_transforms.warps.vertical_warp import (
    VerticalWarp,
)
from cropgen.transforms.intraparagraph_transforms.warps.horizontal_warp import (
    HorizontalWarp,
)
from cropgen.transforms.intraparagraph_transforms.paragraph_tilt import ParagraphTilt
from cropgen.ocrdataset.ocrdataset import OCRDataset
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
import pytest
from PIL import Image
from tqdm.auto import tqdm


def test_ocrdataset(paths: PathBundle, patch_synthetic_manuscript):
    lsi: LabelStudioInterface = paths.lsi  # ty: ignore[invalid-assignment]

    annotations = lsi.get_annotated_pages(use_cache=True, show_progress=True)

    A, B = OCRDataset.from_split(annotations, p=0.95, orders=[2])

    for sample_i in tqdm(range(len(A)), desc="Checking every single sample"):
        sample = A[sample_i]
        assert isinstance(sample["image"], Image.Image)
        assert isinstance(sample["text"], str)
        assert isinstance(sample["sindex"], int)
        assert isinstance(sample["context"], str)
        assert isinstance(sample["order"], (int, str))
        assert isinstance(sample["id"], str)
        assert isinstance(sample["page_id"], (int, str))


def test_segmentationdataset(paths: PathBundle):
    # TODO: remove the autouse from patch image processing
    lsi: LabelStudioInterface = paths.lsi  # ty: ignore[invalid-assignment]

    annotations = lsi.get_annotated_pages(use_cache=True, show_progress=True)

    A, B = SegmentationDataset.from_split(annotations, p=0.95, orders=[2])

    for sample_i in tqdm(range(len(A)), desc="Checking every single sample"):
        sample = A[sample_i]
        assert isinstance(sample[0], Image.Image)
        assert isinstance(sample[1], list)
        assert len(sample[1]) == 2
        assert all(isinstance(pol, Polygon) for pol in sample[1])


# def test_transforms(paths: PathBundle):
#     transforms = [
#         HorizontalWarp(100),
#         HorizontalWarp(0),
#         HorizontalWarp(-100),
#         VerticalWarp(100),
#         VerticalWarp(0),
#         VerticalWarp(-100),
#         ParagraphLinewiseRotation(-30),
#         ParagraphLinewiseRotation(0),
#         ParagraphLinewiseRotation(30),
#         HorizontalMovement("linear", intercept=10, slope=-0.1),
#         HorizontalMovement("linear", intercept=-10, slope=0.1),
#         HorizontalMovement("wave", amplitude=50, period=25),
#         HorizontalMovement("zigzag", amplitude=50),
#         HorizontalMovement("from_amplitude_parameter", amplitude=25),
#         HorizontalMovement("from_amplitude_parameter", amplitude=-25),
#         ParagraphTilt(0.1, tilt_axis="vertical"),
#         ParagraphTilt(-0.1, tilt_axis="vertical"),
#         ParagraphTilt(0.1, tilt_axis="horizontal"),
#         ParagraphTilt(-0.1, tilt_axis="horizontal"),
#     ]
#     lsi: LabelStudioInterface = paths.lsi  # ty: ignore[invalid-assignment]

#     annotations = lsi.get_annotated_pages()
#     dataset = OCRDataset(annotations, orders=[1, 2])
#     indices = [1, 10, 100, 200]
#     for transform in transforms:
#         transform_pack = OCROnTheFlyTransformPack(avoid_intersections=True)
#         transform_pack.add_transform(transform)
#         dataset.set_transform(transform_pack)
#         for idx in indices:
#             dataset[idx]
