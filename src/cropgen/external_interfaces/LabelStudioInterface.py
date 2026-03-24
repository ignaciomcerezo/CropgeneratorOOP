from label_studio_sdk import Client
import json
import os
from src.cropgen.shared.PathBundle import PathBundle
from src.cropgen.external_interfaces.simplify_export import (
    simplify_export,
    load_simplified_export,
)


class LabelStudioInterface:
    slots = (
        "project",
        "local_last_update",
        "usernames",
        "raw_export_path",
        "simplified_filepath",
        "__raw_tasks",
        "__simplified_tasks",
    )

    def __init__(self, paths: PathBundle):
        self.project = None
        self.raw_export_path = paths.raw_export_filepath
        self.simplified_filepath = paths.simplified_filepath

        if not self.raw_export_path.exists():
            raise FileNotFoundError(
                f"No existe export local en {self.raw_export_path}. "
                "Ejecuta primero LabelStudioInterface.update_conditional(paths)."
            )

        loaded_export = list(
            json.loads(self.raw_export_path.read_text(encoding="utf-8"))
        )
        loaded_export.sort(key=lambda tsk: int(tsk["id"]))
        self.__raw_tasks = loaded_export

        if loaded_export:
            self.local_last_update = max(
                task.get("updated_at", "") for task in loaded_export
            )
        else:
            self.local_last_update = None

        self.__simplified_tasks = load_simplified_export(paths)

        if paths.usernames_filepath.exists():
            self.usernames = json.loads(
                paths.usernames_filepath.read_text(encoding="utf-8")
            )
        else:
            self.usernames = []

    @staticmethod
    def _get_latest_update_of_project(project) -> str:
        most_recently_updated_task = project.get_paginated_tasks(
            ordering=["-updated_at"], page=1, page_size=1
        )["tasks"][0]
        return most_recently_updated_task["updated_at"]

    @staticmethod
    def update_conditional(
        paths: PathBundle,
        ls_url: str = None,
        token: str | None = None,
        project_id: int = 4,
        forced: bool = False,
    ) -> bool:

        if not token:
            if "LS_TOKEN" in os.environ:
                token = os.getenv("LS_TOKEN")
            else:
                raise ValueError(
                    "O bien se pasa un token o bien se añade como variable de entorno."
                )

        if not ls_url:
            if "LS_URL" in os.environ:
                ls_url = os.getenv("LS_URL")
            else:
                raise ValueError(
                    "O bien se pasa un ls_url o bien se añade como variable de entorno."
                )

        ls_client = Client(url=ls_url, api_key=token)
        project = ls_client.get_project(id=project_id)

        users = ls_client.get_users()
        user_ids = [user.id for user in users]
        ordered_usernames = []
        if user_ids:
            for x in range(max(user_ids) + 1):
                if x in user_ids:
                    ordered_usernames.append(
                        [u.username for u in users if u.id == x][0]
                    )
                else:
                    ordered_usernames.append(0)
        paths.usernames_filepath.write_text(
            json.dumps(ordered_usernames), encoding="utf-8"
        )

        # comprobamos si hace falta actualizar
        latest_update = LabelStudioInterface._get_latest_update_of_project(project)

        raw_exists = paths.raw_export_filepath.exists()
        simplified_exists = paths.simplified_filepath.exists()

        if raw_exists and not forced:
            loaded_export = list(
                json.loads(paths.raw_export_filepath.read_text(encoding="utf-8"))
            )
            if loaded_export:
                local_last_update = max(
                    task.get("updated_at", "") for task in loaded_export
                )
                if (latest_update <= local_last_update) and simplified_exists:
                    print("Export local ya actualizado. No se descarga nada.")
                    return False

        # descargamos y guardamos el raw
        print("Actualizando export desde Label Studio...")
        raw_tasks = sorted(
            project.export_tasks().copy(), key=lambda tsk: int(tsk["id"])
        )
        paths.raw_export_filepath.write_text(json.dumps(raw_tasks), encoding="utf-8")

        # regeneramos el simplified_tasks
        simplify_export(paths.raw_export_filepath, paths.simplified_filepath)
        simplified_tasks = load_simplified_export(paths)
        paths.simplified_filepath.write_text(
            json.dumps(simplified_tasks), encoding="utf-8"
        )

        return True

    @property
    def raw_tasks(self) -> list:
        return self.__raw_tasks

    @property
    def simplified_tasks(self) -> list:
        return self.__simplified_tasks

    def users(self) -> list[str]:
        return self.usernames

    @property
    def annotations(self) -> list[dict]:
        return [
            r["annotations"][i]
            for r in self.simplified_tasks
            for i in range(len(r["annotations"]))
        ]

    def save_raw_export(self):
        self.raw_export_path.write_text(json.dumps(self.__raw_tasks), encoding="utf-8")

    def save_simplified_export(self):
        self.simplified_filepath.write_text(
            json.dumps(self.__simplified_tasks), encoding="utf-8"
        )

    def __getitem__(self, index: int | str) -> list[dict]:
        if isinstance(index, str):
            index = int(index)
        if not isinstance(index, (str, int)):
            raise TypeError("El índice debe ser entero o string convertible a entero.")

        items = []
        for tsk in self.__simplified_tasks:
            if int(tsk["id"]) > index:
                return items
            elif tsk["id"] == index:
                items.extend(tsk["annotations"])
        return items
