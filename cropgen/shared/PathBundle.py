import urllib.parse
from pathlib import Path
from os import getcwd
import shutil

from cropgen.shared.LSTypedDicts.aggregates import LabelStudioTask
from cropgen.shared.LSTypedDicts.simplified import SimplifiedTask, SimplifiedAnnotation

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
        self.lsi = None
        self.obi = None
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
    def data_out_path(self) -> Path:
        return self.root / "data_out"

    @property
    def crops_path(self) -> Path:
        return self.data_out_path / "crops"

    @property
    def json_filepath(self) -> Path:
        return self.data_out_path / _output_json_filename

    def all_paths(self):
        return [
            self.raw_images_path,
            self.stroke_images_path,
            self.background_images_path,
            self.transcriptions_path,
            self.exports_path,
            self.data_out_path,
            self.crops_path,
            self.data_in_path,
        ]

    def __repr__(self):
        return str(f"<PathBundle with root {self.root}>")

    def assert_paths(self) -> None:
        """
        Comprueba que todas las rutas son accesibles, son instancias de Path válidas y las crea.
        """
        try:
            for path in self.all_paths():
                assert isinstance(path, Path)
                path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                "Error al crear las carpetas necesarias. Revisa que el path raíz es correcto y que tienes permisos de "
                "escritura en esa ubicación."
            )
        except Exception as e:
            raise Exception(f"Error al crear las carpetas necesarias: {e}")

    @staticmethod
    def _simplified_or_raw(
        obj: dict | SimplifiedTask | LabelStudioTask,
    ) -> SimplifiedTask | LabelStudioTask:

        if isinstance(obj, dict):
            try:
                converted_obj = SimplifiedTask.model_validate(obj)
                return converted_obj
            except:
                try:
                    converted_obj = LabelStudioTask.model_validate(obj)
                    return converted_obj
                except:
                    raise TypeError(
                        "Se ha pasado un objeto que no cumple ninguna de las dos."
                    )
        elif isinstance(obj, (SimplifiedTask, LabelStudioTask)):
            return obj
        else:
            raise TypeError("Se ha pasado un tipo incorrecto")

    def _get_image_stem_from_task(
        self, task: dict | LabelStudioTask | SimplifiedTask
    ) -> str | None:
        task: LabelStudioTask | SimplifiedTask = PathBundle._simplified_or_raw(task)
        data = task.data
        image_url = data.image_url or ""

        if not image_url:
            return None

        clean_url = urllib.parse.unquote(image_url)
        filename = clean_url.split("?")[0].split("/")[-1]
        return Path(filename).stem

    def get_raw_image_path_from_task(
        self, task: LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Returns the local path to the corresponding raw image.
        If it cant find it, returns None.
        """
        stem = self._get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the raw image for task: ", task.id)

        filepath = self.get_raw_image_path(stem)

        if filepath.exists():
            return filepath
        print("Could not find the raw image for task: ", task.id)
        return None

    def get_stroke_image_path_from_task(
        self, task: LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Returns the local path to the corresponding stroke image.
        If it cant find it, returns None.
        """
        stem = self._get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the stroke image for task: ", task.id)

        filepath = self.get_stroke_image_path(stem)

        if filepath.exists():
            return filepath

        print("Could not find the stroke image for task: ", task.id)
        return None

    def get_background_image_path_from_task(
        self, task: LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Returns the local path to the corresponding background image.
        If it cant find it, returns None.
        """
        stem = self._get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the background image for task: ", task.id)

        filepath = self.get_background_image_path(stem)

        if filepath.exists():
            return filepath

        print("Could not find the stroke image for task: ", task.id)
        return None

    def get_order_folder(self, order: str | int) -> Path:
        """
        Devuelve la carpeta para guardar los crops de orden 'order'.
        """
        folder = self.crops_path / str(order)
        folder.mkdir(exist_ok=True)
        return folder

    def remove_all_files(self) -> None:
        """
        Elimina todos los archivos de datos de los que depende la generación (data_in, data_out, exports).
        """
        for path in [
            self.data_in_path,
            self.raw_images_path,
            self.transcriptions_path,
            self.exports_path,
            self.data_out_path,
            self.crops_path,
        ]:
            if path.exists() and path.is_dir():
                print(f"Removing folder {path}")
                shutil.rmtree(path)
            elif path.exists():
                raise ValueError(
                    f"Se esperaba una carpeta pero se encontró un archivo en la ruta: {path}"
                )

    def get_worker_json_filepath(self, worker_id: int | None) -> Path:
        """
        Devuelve la ruta al archivo donde cada subproceso de augment_data_parallel debe guardar sus resultados parciales.
        """
        name = self.json_filepath.stem
        extension = self.json_filepath.suffix

        worker_id_str: str = str(worker_id) if worker_id is not None else ""

        worker_filename = f"{name}_{worker_id_str}{extension}"
        return Path(self.json_filepath.parent / worker_filename)

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

    def clean_output_folder(self) -> None:
        """
        Elimina todos los archivos y carpetas dentro de la carpeta de salida (data_out_path),
        pero no elimina la propia carpeta data_out_path.
        """
        self._empty_folder(self.data_out_path)

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

    def remove_downloaded_image_and_transcription(self, page_name: str) -> None:
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

        transcription_path = self.get_transcription_path(page_name)

        if transcription_path.exists():
            transcription_path.unlink()
            print(f"Removed transcription: {transcription_path}")
        else:
            print(f"Unexisting transcirption: {transcription_path}")

    def has_processed_images(self, task: SimplifiedTask | LabelStudioTask):
        return (self.get_background_image_path_from_task(task) is not None) and (
            self.get_stroke_image_path_from_task(task) is not None
        )

    def get_raw_image_path(self, page_name: str | int) -> Path:
        return self.raw_images_path / (self._normalize_page_name(page_name) + ".png")

    def get_stroke_image_path(self, page_name: str | int) -> Path:
        return self.stroke_images_path / (self._normalize_page_name(page_name) + ".png")

    def get_background_image_path(self, page_name: str | int) -> Path:
        return self.background_images_path / (
            self._normalize_page_name(page_name) + ".png"
        )

    def get_transcription_path(self, page_name: str | int) -> Path:
        """Devuelve la ruta a la transcripción asociada a una página concreta."""
        return self.transcriptions_path / (
            self._normalize_page_name(page_name) + ".txt"
        )

    @staticmethod
    def _normalize_page_name(page_name: str | int) -> str:
        """Normaliza el nombre de la página para cuadrar con los usados en el resto del código."""
        page_name: str = str(page_name)
        if (".png" == page_name[-4:]) or (".txt" == page_name[-4:]):
            page_name = page_name[:-4]

        return page_name
