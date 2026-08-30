import urllib.parse
from pathlib import Path
from os import getcwd
import shutil

_raw_export_json_filename = "raw_export.json"
_simplified_export_json_filename = "simplified_export.json"
_output_json_filename = "pairs.jsonl"
_usernames_filename = "usernames.txt"


class PathBundle:
    """
    Clase para almacenar las rutas empleadas durante la generación, y algunas funcionalidades útiles relacionadas
    con los archivos y esta carpeta.
    """

    def __init__(self, root: Path | str | None = None):
        self.root: Path = Path(root) if root else Path(getcwd())
        self.assert_paths()

    @property
    def data_in_path(self) -> Path:
        return self.root / "data_in/"

    @property
    def raw_images_path(self) -> Path:
        return self.data_in_path / "images/raw/"

    @property
    def stroke_images_path(self) -> Path:
        return self.data_in_path / "images/stroke/"

    @property
    def background_images_path(self) -> Path:
        return self.data_in_path / "images/background/"

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
        return self.exports_path / _usernames_filename

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
        Comprueba que todas las rutas son accesibles, son instancias de Path válidas y las crea.
        """
        try:
            for path in self.all_dirs():
                assert isinstance(path, Path)
                path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                "Error al crear las carpetas necesarias. Revisa que el path raíz es correcto y que tienes permisos de "
                "escritura en esa ubicación."
            )
        except Exception as e:
            raise Exception(f"Error al crear las carpetas necesarias: {e}")

    # TODO: separate the task logic from the pathbundle logic

    def remove_all_files(self) -> None:
        """
        Elimina todos los archivos de datos de los que depende la generación (data_in, data_out, exports).
        """
        for path in self.all_dirs():
            if path.exists() and path.is_dir():
                print(f"Removing folder {path}")
                shutil.rmtree(path)
            elif path.exists():
                raise ValueError(f"A folder was expected, but found a file in {path}")

    @staticmethod
    def _empty_folder(folder):
        """Vacía una carpeta indicada."""
        print(f"PathBundle - removing folder <{folder}>")
        if folder.exists() and folder.is_dir():
            for item in folder.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def clean_input_folder(self) -> None:
        """
        Elimina todos los archivos y carpetas dentro de la carpeta de entrada,
        pero no elimina la propia carpeta data_out_path.
        """
        self._empty_folder(self.data_in_path)

    def clean_export_folder(self) -> None:
        """
        Elimina todos los archivos y carpetas dentro de la carpeta de los exports,
        pero no elimina la propia carpeta data_out_path.
        """
        self._empty_folder(self.exports_path)

    def remove_downloaded_image(self, page_name: str) -> None:
        """
        Elimina la imagen y la transcripción asociadas a un nombre de página dado.
        """

        paths = (
            self.get_raw_image_path(page_name),
            self.get_stroke_image_path(page_name),
            self.get_background_image_path(page_name),
        )

        for path in paths:
            if path.exists():
                path.unlink()
                print(f"Remove image: {path}")
            else:
                print(f"Unexisting file: {path}")

    def has_processed_images(self, page_name: str):

        return (
            self.get_background_image_path(page_name).exists()
            and self.get_stroke_image_path(page_name).exists()
        )

    def get_raw_image_path(self, page_name: str | int) -> Path:
        return self.raw_images_path / (self._normalize_page_name(page_name) + ".png")

    def get_stroke_image_path(self, page_name: str | int) -> Path:
        return self.stroke_images_path / (self._normalize_page_name(page_name) + ".png")

    def get_background_image_path(self, page_name: str | int) -> Path:
        return self.background_images_path / (
            self._normalize_page_name(page_name) + ".png"
        )

    @staticmethod
    def _normalize_page_name(page_name: str | int) -> str:
        """Normaliza el nombre de la página para cuadrar con los usados en el resto del código."""
        page_name: str = str(page_name)
        if (".png" == page_name[-4:]) or (".txt" == page_name[-4:]):
            page_name = page_name[:-4]

        return page_name
