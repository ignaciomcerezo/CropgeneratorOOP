import urllib.parse
from pathlib import Path
from os import getcwd
import shutil


class PathBundle:
    def __init__(self, root: Path | str | None = None):
        self.root: Path = Path(root) if root else Path(getcwd())

        self.data_in_path: Path = self.root / "data_in/"
        self.images_path: Path = self.data_in_path / "images/"
        self.transcriptions_path: Path = self.data_in_path / "transcripciones/"

        # carpetas de los exports
        self.exports_path: Path = self.data_in_path / "exports"
        self.raw_export_filepath: Path = self.exports_path / "raw_export.json"
        self.simplified_filepath: Path = self.exports_path / "simplified_export.json"
        self.usernames_filepath: Path = self.exports_path / "usernames.txt"

        # carpetas donde se van a colocar los datos generados.
        self.output_path: Path = self.root / "data_out"
        self.crops_path: Path = self.output_path / "crops"

        try:
            self.assert_paths()
        except PermissionError:
            raise PermissionError(
                "Error al crear las carpetas necesarias. Revisa que el path raíz es correcto y que tienes permisos de escritura en esa ubicación."
            )
        except Exception as e:
            raise Exception(f"Error al crear las carpetas necesarias: {e}")

    def assert_paths(self):
        for path in [
            self.images_path,
            self.transcriptions_path,
            self.exports_path,
            self.output_path,
            self.crops_path,
            self.data_in_path,
        ]:
            assert isinstance(path, Path)
            path.mkdir(parents=True, exist_ok=True)

    def get_image_path_from_task(self, task: dict):
        """
        Dada una tarea, devuelve la ruta LOCAL de la imagen correspondiente.
        Si no se encuentra, devuelve None.
        """
        data = task.get("data", {})
        image_url = data.get("image_url") or data.get("image") or ""

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

        print("No se encontró la imagen para la tarea:", task.get("id"))
        return None

    def remove_all_files(self):
        for path in [
            self.data_in_path,
            self.images_path,
            self.transcriptions_path,
            self.exports_path,
            self.output_path,
            self.crops_path,
        ]:
            if path.exists() and path.is_dir():
                print(f"Removing folder {path}")
                shutil.rmtree(path)
            elif path.exists():
                raise ValueError(
                    f"Se esperaba una carpeta pero se encontró un archivo en la ruta: {path}"
                )
