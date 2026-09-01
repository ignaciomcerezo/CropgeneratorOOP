from typing import Literal
import urllib.parse
from pathlib import Path
from os import getcwd
import shutil

_raw_export_json_filename = "raw_export.json"
_simplified_export_json_filename = "simplified_export.json"


class PathBundle:
    """
    Class used to store all paths used during the structuring, preprocessing and usage of the OCR
    dataset.
    """

    def __init__(self, root: Path | str | None = None):
        self.root: Path = Path(root) if root else Path(getcwd())
        self.assert_paths()

    @property
    def data_in_path(self) -> Path:
        return self.root / "data_in/"

    @property
    def raw_images_path(self) -> Path:
        return self.data_in_path / f"images/raw/"

    @property
    def stroke_images_path(self) -> Path:
        return self.data_in_path / f"images/stroke/"

    @property
    def background_images_path(self) -> Path:
        return self.data_in_path / f"images/background/"

    @property
    def transcriptions_path(self) -> Path:
        return self.data_in_path / "transcriptions/"

    @property
    def exports_path(self) -> Path:
        return self.data_in_path / "exports/"

    @property
    def raw_export_filepath(self) -> Path:
        return self.exports_path / _raw_export_json_filename

    @property
    def simplified_filepath(self) -> Path:
        return self.exports_path / _simplified_export_json_filename

    @property
    def usernames_filepath(self) -> Path:
        return self.exports_path / "usernames.txt"

    @property
    def transcription_path(self) -> Path:
        return self.data_in_path / "transcriptions/"

    @property
    def polygons_path(self) -> Path:
        return self.data_in_path / "polygons/"

    @property
    def metadata_path(self) -> Path:
        return self.data_in_path / "metadata/"

    @property
    def rotations_path(self) -> Path:
        return self.data_in_path / "rotations/"

    @property
    def ids_path(self) -> Path:
        return self.data_in_path / "ids/"

    @staticmethod
    def change_image_category_path(
        image_path: Path, destination_category: Literal["raw", "stroke", "background"]
    ):
        """
        Gets another the corresponding image from a different category. For example, from a background image,
        with destination_category='raw', gets the corresponding raw image.
        """
        if destination_category not in ["raw", "background", "stroke"]:
            raise ValueError(
                f"Image destionation invalid: must be 'raw', 'stroke' or 'background', got {destination_category}"
            )
        return (
            image_path.parents[1]
            / destination_category
            / (image_path.stem + image_path.suffix)
        )

    def all_dirs(self):
        return [
            self.raw_images_path,
            self.stroke_images_path,
            self.background_images_path,
            self.exports_path,
            self.ids_path,
            self.rotations_path,
            self.transcription_path,
            self.polygons_path,
            self.metadata_path,
            self.data_in_path,
        ]

    def __repr__(self):
        return str(f"<PathBundle with root {self.root}>")

    def assert_paths(self) -> None:
        """
        Checks that all paths are accessibles and exist.
        """
        try:
            for path in self.all_dirs():
                assert isinstance(path, Path)
                path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                "Error creating the folders. Check the project root is correct and sufficient permissions have been granted."
                f"\nRoot folder: {self.root}"
            )
        except Exception as e:
            raise Exception(
                f"Unexpected exception encountered while creating PathBundle's folders: {e}"
            )

    def remove_all_files(self) -> None:
        """
        Removes all of the files managed by the project (data_in).
        """
        for path in self.all_dirs():
            if path.exists() and path.is_dir():
                print(f"Removing folder {path}")
                shutil.rmtree(path)
            elif path.exists():
                raise ValueError(f"A folder was expected, but found a file in {path}")

    def remove_downloaded_image(
        self,
        page_name: str,
        image_folder: Literal["raw", "stroke", "background"] = "raw",
    ) -> None:
        """
        Removes the specified image.
        """

        match image_folder.lower():
            case "raw":
                path = self.get_raw_image_path(page_name)
            case "stroke":
                path = self.get_stroke_image_path(page_name)
            case "background":
                path = self.get_background_image_path(page_name)
            case _:
                raise ValueError(
                    f"Unrecognised {image_folder=}. Only raw, stroke and background are accepted"
                )

        if path.exists():
            path.unlink()
            print(f"Removed image stored at {path}.")
        else:
            print(f"No image stored at {path} - no need to remove it.")

    def has_processed_images(self, page_name: str):
        return (
            self.get_background_image_path(page_name).exists()
            and self.get_stroke_image_path(page_name).exists()
        )

    def get_raw_image_path(self, page_name: str | int) -> Path:
        return self.raw_images_path / (str(page_name) + ".png")

    def get_stroke_image_path(self, page_name: str | int) -> Path:
        return self.stroke_images_path / (str(page_name) + ".png")

    def get_background_image_path(self, page_name: str | int) -> Path:
        return self.background_images_path / (str(page_name) + ".png")
