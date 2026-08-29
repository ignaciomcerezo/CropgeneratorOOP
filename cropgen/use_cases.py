import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
import json
from huggingface_hub import HfApi
from PIL import Image
from cropgen.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from tqdm.auto import tqdm

from cropgen.external_interfaces.label_studio.label_studio_interface import (
    LabelStudioInterface,
)
from cropgen.external_interfaces.online_bucket_interface import OnlineBucketInterface
from cropgen.shared.PathBundle import PathBundle
from cropgen.shared.image_processing import separate_background_and_stroke


def setup(
    path: str | Path | None = None,
    obi: OnlineBucketInterface | None = None,
    lsi: LabelStudioInterface | None = None,
    online: bool = True,
    project_id: int = 4,
    bucket_url: str | None = None,
    ls_server_url: str | None = None,
    ls_token: str | None = None,
) -> PathBundle:
    """
    Descarga todos los archivos necesarios para crear el conjunto de datos, y genera sus respectivas interfaces.
    """
    load_dotenv()
    path: Path = Path(path) if path is not None else Path(os.getcwd())
    paths = PathBundle(path)

    obi: OnlineBucketInterface = (
        OnlineBucketInterface.from_env(paths, online=online, bucket_url=bucket_url)
        if obi is None
        else obi
    )
    if online:
        obi.update()

    lsi = (
        LabelStudioInterface.from_env(
            paths, online, project_id, ls_server_url=ls_server_url, ls_token=ls_token
        )
        if lsi is None
        else lsi
    )

    paths.lsi = lsi
    paths.obi = obi

    for task in tqdm(
        [task for task in lsi.simplified_tasks if not paths.has_processed_images(task)],
        desc="Stroke/background separation.",
    ):
        raw_image_path = paths.get_raw_image_path_from_task(task)

        if raw_image_path is None:
            raise ValueError(
                f"Internal error. Did not download all appropriate images: {raw_image_path}"
            )

        raw_image = Image.open(raw_image_path)

        background, stroke = separate_background_and_stroke(
            raw_image,
            out_longest_side=DATASET_LONGEST_SIZE_PX,
            processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
        )
        stroke.save(paths.stroke_images_path / f"{raw_image_path.stem}.png")
        background.save(paths.background_images_path / f"{raw_image_path.stem}.png")

    return paths
