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

        self.data_in_path: Path = self.root / "data_in/"
        self.images_path: Path = self.data_in_path / "images/"
        self.transcriptions_path: Path = self.data_in_path / "transcripciones/"

        # carpetas de los exports
        self.exports_path: Path = self.data_in_path / "exports"
        self.raw_export_filepath: Path = self.exports_path / _raw_export_json_filename
        self.simplified_filepath: Path = (
            self.exports_path / _simplified_export_json_filename
        )
        self.usernames_filepath: Path = self.exports_path / _usernames_filename

        # carpetas donde se van a colocar los datos generados.
        self.data_out_path: Path = self.root / "data_out"
        self.crops_path: Path = self.data_out_path / "crops"
        self.json_filepath: Path = self.data_out_path / _output_json_filename

        self.dataset_path = self.root / "dataset"

        try:
            self.assert_paths()
        except PermissionError:
            raise PermissionError(
                "Error al crear las carpetas necesarias. Revisa que el path raíz es correcto y que tienes permisos de "
                "escritura en esa ubicación."
            )
        except Exception as e:
            raise Exception(f"Error al crear las carpetas necesarias: {e}")

    def __repr__(self):
        return str(f"<PathBundle con raíz {self.root}>")

    def assert_paths(self) -> None:
        """
        Comprueba que todas las rutas son accesibles, son instancias de Path válidas y las crea.
        """
        for path in [
            self.images_path,
            self.transcriptions_path,
            self.exports_path,
            self.data_out_path,
            self.crops_path,
            self.data_in_path,
        ]:
            assert isinstance(path, Path)
            path.mkdir(parents=True, exist_ok=True)

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

    def get_image_path_from_task(
        self, task: dict | LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Dada una tarea, devuelve la ruta LOCAL de la imagen correspondiente.
        Si no se encuentra, devuelve None.
        """

        task: LabelStudioTask | SimplifiedTask = PathBundle._simplified_or_raw(task)
        data = task.data
        image_url = data.image_url or ""

        if not image_url:
            return None

        clean_url = urllib.parse.unquote(image_url)
        filename = clean_url.split("?")[0].split("/")[-1]

        exact_path = self.images_path / filename
        if exact_path.exists():
            return exact_path

        stem = Path(filename).stem
        for p in self.images_path.iterdir():
            if p.is_file() and (p.suffix.lower() == ".png"):
                if p.stem == stem:
                    return p

        print("No se encontró la imagen para la tarea:", task.id)
        return None

    def get_order_folder(self, order: str | int) -> Path:
        """
        Devuelve la carpeta para guardar los crops de orden 'order'.
        """
        folder = self.crops_path / str(order)
        if not folder.exists():
            folder.mkdir()
        return folder

    def remove_all_files(self) -> None:
        """
        Elimina todos los archivos de datos de los que depende la generación (data_in, data_out, exports).
        """
        for path in [
            self.data_in_path,
            self.images_path,
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
        image_path = self.get_image_path(page_name)
        transcription_path = self.get_transcription_path(page_name)

        if image_path.exists():
            image_path.unlink()
            print(f"Imagen eliminada: {image_path}")
        else:
            print(f"No se encontró la imagen: {image_path}")

        if transcription_path.exists():
            transcription_path.unlink()
            print(f"Transcripción eliminada: {transcription_path}")
        else:
            print(f"No se encontró la transcripción: {transcription_path}")

    def get_image_path(self, page_name: str | int) -> Path:
        """Devuelve la ruta a la imagen asociada a una página concreta."""
        return self.images_path / (self._normalize_page_name(page_name) + ".png")

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

        if len(page_name) < 3 and page_name.isdigit():
            page_name = page_name.rjust(3, "0")
        return page_name
