from cropgen.shared.page_metadata import PageSampleMetadata
from cropgen.external_interfaces.external_interface import ExternalInterface
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from cropgen.shared.path_bundle import PathBundle


def setup(
    paths: PathBundle,
    external_interfaces: list[ExternalInterface],
) -> PathBundle:
    """
    Descarga todos los archivos necesarios para crear el conjunto de datos, y genera sus respectivas interfaces.
    """
    load_dotenv()

    for external_interface in external_interfaces:
        external_interface.setup()

    return paths
