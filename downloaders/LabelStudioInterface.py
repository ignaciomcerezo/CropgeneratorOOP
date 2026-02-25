from label_studio_sdk import Client
from LS_token import ls_token
from paths import LS_url, raw_export_filepath
import json

class LabelStudioInterface:
    slots = ("project", "__tasks", "local_last_update", "usernames", "export_path")



    def __init__(self, token:str=None, project_id=4, ls_url = LS_url, export_path = raw_export_filepath):
        """
        Interfaz con La
        :param token:
        :param project_id:
        :param ls_url:
        :param export_path:
        """
        token = token or ls_token

        ls_client = Client(url=ls_url, api_key=token)
        self.project = ls_client.get_project(id=project_id)

        users = ls_client.get_users()

        user_ids = [user.id for user in users]

        self.export_path = export_path



        ordered_usernames = []
        for x in range(max(user_ids) + 1):
            if x in user_ids:
                ordered_usernames.append(
                    [user.username for user in users if user.id == x][0]
                )
            else:
                ordered_usernames.append(0)
        self.usernames = ordered_usernames

        self.__tasks = None
        self.local_last_update = None

        if self.export_path.exists():
            loaded_export = json.loads(self.export_path.read_text())

            loaded_export_last_updated_at = sorted([task["updated_at"] for task in loaded_export])[-1]
            last_update = self._get_latest_update_of_LS()

            if last_update > loaded_export_last_updated_at:
                self._update_tasks_conditional(forced = True)
                self.save_export()
            else:
                self.__tasks = loaded_export
                self.local_last_update = loaded_export_last_updated_at


        else:
            self._update_tasks_conditional(forced = True)



    @property
    def tasks(self, check_updated:bool = True) -> dict:
        """
        Devuelve las tareas ya etiquetadas del servidor de LabelStudio al que se está accediendo.
        Si check_updated, se comprueba primero que estén actualizadas comparadas con las del servidor.
        """
        if check_updated:
            self._update_tasks_conditional()
        return self.__tasks


    def users(self) -> list["str"]:
        """
        Lista de nombres de usuario, ordenados según el orden interno de LabelStudio.
        Si se ha borrado un usuario, su nombre se sustituye con 0
        """
        return self.usernames

    def _get_latest_update_of_LS(self):
        most_recently_updated_task = self.project.get_paginated_tasks(
            ordering=['-updated_at'],
            page=1,
            page_size=1
        )["tasks"][0]
        update_date = most_recently_updated_task["updated_at"]
        return update_date

    def _update_tasks_conditional(self, forced = False):
        try:
            latest_update_of_LS = self._get_latest_update_of_LS()
            if forced or (not self.local_last_update or  (latest_update_of_LS > self.local_last_update)):
                #cargamos el export
                self.__tasks = self.project.export_tasks()
                self.local_last_update = latest_update_of_LS

                self.save_export()
        except Exception as e:
            print("Ha ocurrido un error durante la actualización de las tareas de LabelStudioInterface.")
            raise e

    def save_export(self):
        self.export_path.write_text(json.dumps(self.__tasks))


