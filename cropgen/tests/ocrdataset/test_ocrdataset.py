from cropgen.ocrdataset.ocrdataset import OCRDataset
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.shared.PathBundle import PathBundle
import pytest
from PIL import Image
from tqdm.auto import tqdm


def test_ocrdataset(paths: PathBundle, patch_synthetic_manuscript):
    lsi: LabelStudioInterface = paths.lsi  # ty: ignore[invalid-assignment]

    annotations = lsi.get_annotated_pages(use_cache=True, show_progress=True)

    A, B = OCRDataset.from_split(annotations, p=0.95, orders=[1])

    for sample_i in tqdm(range(len(A)), desc="Checking every single sample"):
        sample = A[sample_i]
        assert isinstance(sample["image"], Image.Image)
        assert isinstance(sample["text"], str)
        assert isinstance(sample["sindex"], int)
        assert isinstance(sample["context"], str)
        assert isinstance(sample["order"], (int, str))
        assert isinstance(sample["id"], str)
        assert isinstance(sample["page_id"], (int, str))
