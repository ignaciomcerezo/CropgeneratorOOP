import json
import os
from pathlib import Path
from urllib.parse import unquote as url_unquote

from label_studio_sdk import Client
from shapely.geometry import Polygon

from cropgen.external_interfaces.external_interface import ExternalInterface
from cropgen.external_interfaces.label_studio.helpers.json_conversor import (
    extract_bounds,
    pair_lines,
)
from cropgen.external_interfaces.label_studio.helpers.simplify_export import (
    simplify_tasks,
)
from cropgen.external_interfaces.label_studio.ls_typed_dicts import (
    RectangleResult,
    SimplifiedAnnotation,
    SimplifiedTask,
)
from cropgen.shared.geometry_processing import calculate_reading_angle
from cropgen.shared.path_bundle import PathBundle


class _LSUsersManager:
    def __init__(self):
        self.usernames: list[str] | None = None

    def __getitem__(self, index):
        if self.usernames is None:
            return "Offline/Unknown"
        elif index < len(self.usernames):
            return self.usernames[index]
        else:
            return "Impossible username"

    def update_usernames(self, usernames: list[str]):
        self.usernames = usernames

    def __repr__(self):
        return f"<_LSUsersManager with usernames={self.usernames}.>"


class LabelStudioInterface(ExternalInterface):
    """Manages Label Studio integration by pulling remote tasks and
    transforming them directly into serialized dataset artifacts.
    """

    slots = (
        "project_id",
        "server_url",
        "token",
        "online",
        "paths",
        "usernames",
    )

    def __init__(
        self,
        server_url: str,
        token: str,
        project_id: int = 4,
        online: bool = True,
    ):
        self.online = online
        self.project_id = project_id
        self.token = token
        self.url = server_url
        self._usernames: _LSUsersManager = _LSUsersManager()

    @classmethod
    def from_env(
        cls,
        online: bool = True,
        project_id: int = 4,
        ls_token: str | None = None,
        ls_server_url: str | None = None,
        token_env_var: str = "LS_TOKEN",
        url_env_var: str = "LS_URL",
    ) -> "LabelStudioInterface":
        if (token_env_var not in os.environ) and (ls_token is None):
            raise ValueError(
                f"{token_env_var} no está presente en las variables de entorno."
            )
        if (url_env_var not in os.environ) and (ls_server_url is None):
            raise ValueError(
                f"{url_env_var} no está presente en las variables de entorno."
            )

        token = ls_token if ls_token is not None else str(os.getenv(token_env_var))
        url = (
            ls_server_url if ls_server_url is not None else str(os.getenv(url_env_var))
        )

        obj = cls(url, token, project_id, online)
        return obj

    def __repr__(self):
        return f"<LabelStudioInterface with URL {self.url}>"

    def test_connection_successful(self) -> bool:
        try:
            client = Client(url=self.url, api_key=self.token)
            client.check_connection()
            print("LSI Connection successful.")
            return True
        except Exception:
            print("LSI Connection unsuccessful.")
            return False

    def _update_usernames(self, ls_client: Client | None = None) -> None:
        if ls_client is None:
            ls_client = Client(url=self.url, api_key=self.token)

        users = ls_client.get_users()
        user_ids = [user.id for user in users]
        ordered_usernames: list[str] = []
        if user_ids:
            for x in range(max(user_ids) + 1):
                matching = [u.username for u in users if u.id == x]
                if matching:
                    ordered_usernames.append(matching[0])
                else:
                    ordered_usernames.append("Impossible LS user")
        self._usernames.update_usernames(ordered_usernames)

    def fetch_simplified_tasks(self) -> list[SimplifiedTask]:
        """Pulls task data from Label Studio and simplifies."""
        if not self.online:
            print(f"LSI configured with online={self.online}; skipping remote fetch.")
            return []

        ls_client = Client(url=self.url, api_key=self.token)
        project = ls_client.get_project(id=self.project_id)

        self._update_usernames()
        print("Downloading and simplifying tasks from Label Studio...")
        raw_tasks_data = project.export_tasks()
        raw_tasks_data.sort(
            key=lambda t: t.get("id", 0) if isinstance(t, dict) else t.id
        )

        return simplify_tasks(raw_tasks_data)

    def users(self) -> _LSUsersManager:
        return self._usernames

    def parts_managed(self):
        return {"metadata", "rotations"}

    def parts_required(self):
        return {"background_images", "stroke_images"}

    def setup(self, paths: PathBundle) -> None:
        """Fetches remote annotations and writes individual transcription,

        polygon, rotation, id, and metadata files.
        """
        if not self.online:
            print(
                f"LSI configured with online={self.online}; keeping local generated data."
            )
            return

        tasks = self.fetch_simplified_tasks()

        for task in tasks:
            image_url = task.data.image_url
            task_id = task.id
            page = Path(url_unquote(image_url)).stem

            for subindex, simplified_ann in enumerate(task.annotations):
                transcriptions: list[str] = []
                poly_coords: list[list[tuple[float, float]]] = []
                ids: list[str] = []
                rotations: list[float] = []

                completer: str = self._usernames[simplified_ann.completed_by]
                updater: str = self._usernames[simplified_ann.updated_by]
                ann_id = simplified_ann.id

                results = simplified_ann.result
                box2text, id2boxres, id2txtres = pair_lines(results)

                trios = [
                    (
                        id2boxres[key],
                        id2txtres[box2text[key]],
                        f"{key}-{box2text[key]}",
                    )
                    for key in id2boxres
                    if key in box2text and box2text[key] in id2txtres
                ]

                for trio in trios:
                    box_result = trio[0]
                    txt_result = trio[1]
                    transcription = txt_result.value.text
                    assert len(transcription) == 1
                    transcriptions.append(transcription[0])

                    poly_bounds = extract_bounds(box_result)
                    assert len(poly_bounds) > 3
                    poly_coords.append(poly_bounds)

                    ids.append(trio[2])

                    if isinstance(box_result, RectangleResult):
                        rotations.append(box_result.value.rotation)
                    else:
                        rotations.append(calculate_reading_angle(Polygon(poly_bounds)))

                json_name = f"s{subindex}_pg{page}.json"

                transcriptions_filepath = paths.transcription_path / json_name
                polygons_filepath = paths.polygons_path / json_name
                rotations_filepath = paths.rotations_path / json_name
                ids_path = paths.ids_path / json_name
                image_path = str(paths.raw_images_path / f"{page}.png")

                transcriptions_filepath.write_text(json.dumps(transcriptions))
                polygons_filepath.write_text(json.dumps(poly_coords))
                rotations_filepath.write_text(json.dumps(rotations))
                ids_path.write_text(json.dumps(ids))

                metadata = {
                    "page": page,
                    "task_id": task_id,
                    "completer": completer,
                    "updater": updater,
                    "subindex": subindex,
                    "ann_id": ann_id,
                    "order": len(trios),
                    "image_path": image_path,
                    "ids_path": str(ids_path),
                    "transcriptions_path": str(transcriptions_filepath),
                    "polygons_path": str(polygons_filepath),
                    "rotations_path": str(rotations_filepath),
                    "polygons_are_in_percentage": True,
                    "source": "Label Studio",
                }

                (paths.metadata_path / json_name).write_text(json.dumps(metadata))
