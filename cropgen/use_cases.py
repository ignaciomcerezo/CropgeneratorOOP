import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
import json
from huggingface_hub import HfApi

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.shared.PathBundle import PathBundle


def setup(
    path: str | Path | None = None,
    obi: OracleBucketInterface | None = None,
    lsi: LabelStudioInterface | None = None,
    online: bool = True,
    project_id: int = 4,
) -> PathBundle:
    """
    Descarga todos los archivos necesarios para crear el conjunto de datos, y genera sus respectivas interfaces.
    """
    load_dotenv()
    path: Path = Path(path) if path is not None else Path(os.getcwd())
    paths = PathBundle(path)

    obi: OracleBucketInterface = (
        OracleBucketInterface.from_env(paths, online=online) if obi is None else obi
    )
    if online:
        obi.update()

    lsi = (
        LabelStudioInterface.from_env(paths, online, project_id) if lsi is None else lsi
    )
    if online:
        lsi.fetch_and_simplify()

    paths.lsi = lsi
    paths.obi = obi

    return paths
