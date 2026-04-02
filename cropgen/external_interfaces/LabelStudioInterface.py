from label_studio_sdk import Client
import json
import os
from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.simplify_export import (
    simplify_export,
    load_simplified_export,
)
from cropgen.shared.LSTypedDicts.aggregates import LabelStudioTask
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedTask,
    SimplifiedAnnotation,
)


class LabelStudioInterface:
    """
    Clase para gestionar la interacción con Label Studio, incluyendo la descarga, actualización y simplificación de exports,
    así como el acceso a tareas, anotaciones y usuarios. Utiliza rutas proporcionadas por un PathBundle.
    """

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
        """
        Inicializa la interfaz de Label Studio a partir de un PathBundle.
        Carga los exports locales (raw y simplified) y la lista de usuarios si existen.
        Lanza un error si no existe el export raw local.
        """
        self.project = None
        self.raw_export_path = paths.raw_export_filepath
        self.simplified_filepath = paths.simplified_filepath

        if not self.raw_export_path.exists():
            raise FileNotFoundError(
                f"No existe export local en {self.raw_export_path}. "
                "Ejecuta primero LabelStudioInterface.update_conditional(paths)."
            )

        loaded_export_dicts = list(
            json.loads(self.raw_export_path.read_text(encoding="utf-8"))
        )
        loaded_export_dicts.sort(key=lambda tsk: int(tsk["id"]))
        # Convertir a instancias de LabelStudioTask
        loaded_export = [
            LabelStudioTask.model_validate(tsk) for tsk in loaded_export_dicts
        ]
        self.__raw_tasks = loaded_export

        if loaded_export:
            self.local_last_update = max(task.updated_at for task in loaded_export)
        else:
            self.local_last_update = None

        # Cargar y convertir simplified_tasks a instancias de SimplifiedTask
        simplified_tasks_dicts = load_simplified_export(paths)
        self.__simplified_tasks = [
            SimplifiedTask.model_validate(tsk) for tsk in simplified_tasks_dicts
        ]

        if paths.usernames_filepath.exists():
            self.usernames = list(
                json.loads(paths.usernames_filepath.read_text(encoding="utf-8"))
            )
        else:
            self.usernames = []

    @staticmethod
    def _get_latest_update_of_project(project) -> str:
        """
        Devuelve la fecha de la última actualización de una tarea en el proyecto de Label Studio.
        """
        most_recently_updated_task = project.get_paginated_tasks(
            ordering=["-updated_at"], page=1, page_size=1
        )["tasks"][0]
        return most_recently_updated_task["updated_at"]

    @staticmethod
    def update_conditional(
        paths: PathBundle,
        ls_url: str | None = None,
        token: str | None = None,
        project_id: int = 4,
        forced: bool = False,
    ) -> bool:
        """
        Actualiza los archivos de exportación desde Label Studio si hay cambios o si se fuerza la actualización.
        Descarga los datos, los guarda y regenera el export simplificado.
        Devuelve True si se ha actualizado, False si ya estaba actualizado.
        """

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
    def raw_tasks(self) -> list[LabelStudioTask]:
        """
        Devuelve la lista de tareas raw descargadas de Label Studio
        """
        return self.__raw_tasks

    @property
    def simplified_tasks(self) -> list[SimplifiedTask]:
        """
        Devuelve la lista de tareas simplificadas a partir del raw export
        """
        return self.__simplified_tasks

    def users(self) -> list[str]:
        """
        Devuelve la lista de nombres de usuario asociados del proyecto de LabelStudio
        """
        return self.usernames

    @property
    def annotations(self) -> list[SimplifiedAnnotation]:
        """
        Devuelve una lista de todas las anotaciones presentes en las tareas simplificadas
        """
        return [
            tsk.annotations[i]
            for tsk in self.simplified_tasks
            for i in range(len(tsk.annotations))
        ]

    def save_raw_export(self):
        """
        Guarda el export raw actual en disco, sobrescribiendo el archivo correspondiente.
        """
        dump_data = [task.model_dump(mode="json") for task in self.__raw_tasks]
        self.raw_export_path.write_text(json.dumps(dump_data), encoding="utf-8")

    def save_simplified_export(self):
        """
        Guarda el export simplificado actual, sobrescribiendo el archivo correspondiente si hubiera había
        """
        dump_data = [task.model_dump(mode="json") for task in self.__simplified_tasks]
        self.simplified_filepath.write_text(json.dumps(dump_data), encoding="utf-8")

    def __getitem__(self, index: int | str) -> list[SimplifiedAnnotation]:
        """
        Permite acceder a las anotaciones de una tarea por su índice (int o str). Devuelve una lista de anotaciones
        para la tarea correspondiente.
        """
        if isinstance(index, str):
            index = int(index)
        if not isinstance(index, (str, int)):
            raise TypeError("El índice debe ser entero o string convertible a entero.")

        items: list[SimplifiedAnnotation] = []
        for tsk in self.__simplified_tasks:
            if int(tsk.id) > index:
                return items
            elif tsk.id == index:
                items.extend(tsk.annotations)
        return items
