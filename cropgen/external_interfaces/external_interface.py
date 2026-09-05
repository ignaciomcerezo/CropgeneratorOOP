from cropgen.shared.path_bundle import PathBundle
from pathlib import Path
from typing import Literal, Collection, Sequence
from abc import ABC, abstractmethod
from cropgen.shared.page_metadata import PageSampleMetadata

_PARTS = set[
    Literal[
        "raw_images",
        "background_images",
        "stroke_images",
        "metadata",
        "polygons",
        "rotations",
        "transcriptions",
    ]
]


class ExternalInterface(ABC):

    @abstractmethod
    def setup(self, paths: PathBundle) -> None:
        """
        Completes the setup related to this external interface's managed data in's parts.
        They are created with the .setup() method.
        """
        raise NotImplementedError

    @abstractmethod
    def parts_required(self) -> _PARTS:
        """
        Returns whose parts of the data_in are required by this external interface's to do its job.
        """
        raise NotImplementedError

    @abstractmethod
    def parts_managed(self) -> _PARTS:
        """
        Returns whose parts of the data_in are created and managed by this external interface's.
        They are created with .setup().
        """
        raise NotImplementedError
