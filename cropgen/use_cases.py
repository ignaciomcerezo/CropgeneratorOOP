from cropgen.shared.page_metadata import PageSampleMetadata
from cropgen.external_interfaces.external_interface import ExternalInterface
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from cropgen.shared.path_bundle import PathBundle


def setup(
    path: str | Path | Literal["cwd"],
    external_interfaces: list[ExternalInterface],
) -> PathBundle:
    """
    Descarga todos los archivos necesarios para crear el conjunto de datos, y genera sus respectivas interfaces.
    """
    load_dotenv()
    path: Path = Path(path) if path is not "cwd" else Path(os.getcwd())
    paths = PathBundle(path)

    for external_interface in external_interfaces:
        external_interface.setup()

    return paths
