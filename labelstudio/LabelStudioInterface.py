from label_studio_sdk import Client
from LS_token import ls_token
from paths import (
    LS_url,
    exports_path,
    raw_export_filepath as default_raw_export_filepath,
    simplified_filepath as default_simplified_export_filepath,
)
import json
from labelstudio.simplify_export import (
    simplify_export,
    load_simplified_export,
)
from preprocessing.helpers.helper_to_classes import get_image_path_from_task
from PIL import Image


class LabelStudioInterface:
    slots = (
        "project",
        "local_last_update",
        "usernames",
        "raw_export_filepath",
        "simplified_export_filepath",
        "__raw_tasks",
        "__simplified_tasks",
    )

    def __init__(
        self,
        token: str = None,
        project_id=4,
        ls_url=LS_url,
        raw_export_filepath=default_raw_export_filepath,
        simplified_export_filepath=default_simplified_export_filepath,
    ):
        """
        Interfaz con La que nos comunicamos con el servidor de LabelStudio. Comprueba de manera inteligente si
        es necesario actualizar las updates.
        """
        token = token or ls_token

        ls_client = Client(url=ls_url, api_key=token)
        self.project = ls_client.get_project(id=project_id)

        users = ls_client.get_users()

        user_ids = [user.id for user in users]

        self.raw_export_path = raw_export_filepath
        self.simplified_filepath = simplified_export_filepath

        ordered_usernames = []
        for x in range(max(user_ids) + 1):
            if x in user_ids:
                ordered_usernames.append(
                    [user.username for user in users if user.id == x][0]
                )
            else:
                ordered_usernames.append(0)
        self.usernames = ordered_usernames

        (exports_path / "usernames.txt").write_text(json.dumps(self.usernames))

        self.__raw_tasks = None
        self.local_last_update = None

        if self.raw_export_path.exists():
            loaded_export = list(json.loads(self.raw_export_path.read_text()))

            loaded_export_last_updated_at = sorted(
                [task["updated_at"] for task in loaded_export]
            )[-1]
            last_update = self._get_latest_update_of_LS()

            if last_update > loaded_export_last_updated_at:
                self._update_tasks_conditional(forced=True)

            else:
                loaded_export.sort(key=lambda tsk: int(tsk["id"]))

                self.__raw_tasks = loaded_export
                self.local_last_update = loaded_export_last_updated_at
                self.__simplified_tasks = load_simplified_export(
                    self.simplified_filepath
                )

        else:
            self._update_tasks_conditional(forced=True)

    @property
    def raw_tasks(self, check_updated: bool = False) -> list:
        """
        Devuelve las tareas ya etiquetadas del servidor de LabelStudio al que se está accediendo.
        Si check_updated, se comprueba primero que estén actualizadas comparadas con las del servidor.
        """
        if check_updated:
            self._update_tasks_conditional()
        return self.__raw_tasks

    @property
    def simplified_tasks(self, check_updated: bool = False) -> list:
        if check_updated and self.is_outdated:
            self._update_tasks_conditional(forced=True)
            self._set_and_save_simplified_tasks()
        return self.__simplified_tasks

    def users(self) -> list["str"]:
        """
        Lista de nombres de usuario, ordenados según el orden interno de LabelStudio.
        Si se ha borrado un usuario, su nombre se sustituye con 0
        """
        return self.usernames

    @property
    def annotations(self):
        return [
            r["annotations"][i]
            for r in self.simplified_tasks
            for i in range(len(r["annotations"]))
        ]

    def _get_latest_update_of_LS(self):
        most_recently_updated_task = self.project.get_paginated_tasks(
            ordering=["-updated_at"], page=1, page_size=1
        )["tasks"][0]
        update_date = most_recently_updated_task["updated_at"]
        return update_date

    @property
    def is_outdated(self):

        return (self.local_last_update is None) or (
            self._get_latest_update_of_LS() > self.local_last_update
        )

    def _update_tasks_conditional(self, forced=False):
        try:
            latest_update_of_LS = self._get_latest_update_of_LS()

            if forced or (not self.raw_export_path.exists()) or self.is_outdated:
                print("Actualizando export (update_tasks_conditional).")
                # cargamos el export

                self.__raw_tasks = sorted(
                    self.project.export_tasks().copy(), key=lambda tsk: int(tsk["id"])
                )
                self.local_last_update = latest_update_of_LS
                self._set_and_save_simplified_tasks()
        except Exception as e:
            print(
                "Ha ocurrido un error durante la actualización de las tareas de LabelStudioInterface."
            )
            raise e

    def save_raw_export(self):
        self.raw_export_path.write_text(json.dumps(self.__raw_tasks))

    def save_simplified_export(self):
        self.simplified_filepath.write_text(json.dumps(self.__simplified_tasks))

    def _set_and_save_simplified_tasks(self) -> None:
        """
        Saves the raw export, simplifies it and saves the simplified tasks.
        """
        self.save_raw_export()
        simplify_export(self.raw_export_path, self.simplified_filepath)
        self.__simplified_tasks = load_simplified_export(self.simplified_filepath)
        self.save_simplified_export()

    def __getitem__(self, index: int | str) -> list[dict]:
        """Devuelve todas las anotaciones de self.simplified_export que tengan ["id"] = index.
        No comprueba que esté actualizado."""
        if isinstance(index, str):
            index = int(index)
        if not isinstance(index, (str, int)):
            raise TypeError(
                "El valor del índice dado a la instancia de LabelStudioInterface debe ser un entero o un string."
            )

        items = []
        for tsk in self.__simplified_tasks:
            if int(tsk["id"]) > index:
                return items
            elif tsk["id"] == index:
                items.extend(tsk["annotations"])
        return items

    def get_image(self, task_id):
        task = [task for task in self.simplified_tasks if task["id"] == int(task_id)][0]
        return Image.open(get_image_path_from_task(task))
