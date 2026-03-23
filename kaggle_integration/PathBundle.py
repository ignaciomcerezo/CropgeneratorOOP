from pathlib import Path
from dataclasses import dataclass
from os import getcwd


class PathBundle:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else getcwd()

        self.data_in_path = self.root / "data_in/"
        self.images_path = self.data_in_path / "images/"
        self.transcriptions_path = self.data_in_path / "transcripciones/"

        # carpetas de los exports
        self.exports_path = self.data_in_path / "exports"
        self.raw_export_filepath = self.exports_path / "raw_export.json"
        self.simplified_filepath = self.exports_path / "simplified_export.json"
        self.usernames_filepath = self.exports_path / "usernames.txt"

        # carpetas donde se van a colocar los datos generados.
        self.output_path = root / "data_out"
        self.crops_path = self.output_path / "crops"

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
            path.mkdir(parents=True, exist_ok=True)
