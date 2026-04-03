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
from pathlib import Path


class LabelStudioInterface:
    """
    Clase para gestionar la interacción con Label Studio, incluyendo la descarga, actualización y simplificación de exports,
    así como el acceso a tareas, anotaciones y usuarios. Utiliza rutas proporcionadas por un PathBundle.
    """

    slots = (
        "project",
        "local_last_update",
        "usernames",
        "raw_export_filepath",
        "simplified_export_filepath",
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
        self.raw_export_filepath = paths.raw_export_filepath
        self.simplified_export_filepath = paths.simplified_filepath

        if not self.raw_export_filepath.exists():
            raise FileNotFoundError(
                f"No existe export local en {self.raw_export_filepath}. "
                "Ejecuta primero LabelStudioInterface.update_conditional(paths)."
            )

        self.__raw_tasks = self._load_raw_as_schema(self.raw_export_filepath)

        self.__raw_tasks.sort(key=lambda task: task.id)

        if self.__raw_tasks:
            self.local_last_update = max(task.updated_at for task in self.__raw_tasks)
        else:
            self.local_last_update = None

        # Cargar y convertir simplified_tasks a instancias de SimplifiedTask
        self.__simplified_tasks = self._load_simplified_as_schema(
            self.simplified_export_filepath
        )
        self.__simplified_tasks.sort(key=lambda task: task.id)

        if paths.usernames_filepath.exists():
            self.usernames: list[str] = list(
                json.loads(paths.usernames_filepath.read_text(encoding="utf-8"))
            )
        else:
            self.usernames: list[str] = []

    @staticmethod
    def _get_latest_update_of_project(project) -> str:
        """
        Devuelve la fecha de la última actualización de una tarea en el proyecto de Label Studio.
        """
        # most_recently_updated_task = LabelStudioTask.model_validate(
        #     project.get_paginated_tasks(ordering=["-updated_at"], page=1, page_size=1)[
        #         "tasks"
        #     ][0]
        # )
        updated_at = project.get_paginated_tasks(
            ordering=["-updated_at"], page=1, page_size=1
        )["tasks"][0]["updated_at"]
        return str(updated_at)

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

        if paths.raw_export_filepath.exists() and not forced:
            loaded_export = LabelStudioInterface._load_raw_as_schema(
                paths.raw_export_filepath
            )

            if loaded_export:
                local_last_update = max(task.updated_at for task in loaded_export)
                if (
                    latest_update <= local_last_update
                ) and paths.simplified_filepath.exists():
                    print("Export local ya actualizado. No se descarga nada.")
                    return False

        # descargamos y guardamos el raw
        print("Actualizando export desde Label Studio...")
        raw_tasks: list[LabelStudioTask] = [
            LabelStudioTask.model_validate(task_dict)
            for task_dict in project.export_tasks().copy()
        ]
        raw_tasks.sort(key=lambda task: task.id)

        dump_data = [task.model_dump(mode="json") for task in raw_tasks]
        paths.raw_export_filepath.write_text(json.dumps(dump_data), encoding="utf-8")

        # regeneramos el simplified_tasks
        simplify_export(paths.raw_export_filepath, paths.simplified_filepath)
        # simplified_tasks: list[SimplifiedTask] = load_simplified_export(paths)
        # paths.simplified_export_filepath.write_text(
        #     json.dumps(simplified_tasks), encoding="utf-8"
        # )

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

    def save_raw_export(self) -> None:
        """
        Guarda el export raw actual en disco, sobrescribiendo el archivo correspondiente.
        """
        dump_data = [task.model_dump(mode="json") for task in self.__raw_tasks]
        self.raw_export_filepath.write_text(json.dumps(dump_data), encoding="utf-8")

    def save_simplified_export(self) -> None:
        """
        Guarda el export simplificado actual, sobrescribiendo el archivo correspondiente si hubiera había
        """
        dump_data = [task.model_dump(mode="json") for task in self.__simplified_tasks]
        self.simplified_export_filepath.write_text(
            json.dumps(dump_data), encoding="utf-8"
        )

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

    @staticmethod
    def _load_raw_as_schema(filepath: Path) -> list[LabelStudioTask]:
        return [
            LabelStudioTask.model_validate(task_dict)
            for task_dict in json.loads(filepath.read_text(encoding="utf-8"))
        ]

    @staticmethod
    def _load_simplified_as_schema(filepath: Path) -> list[SimplifiedTask]:
        return [
            SimplifiedTask.model_validate(task_dict)
            for task_dict in json.loads(filepath.read_text(encoding="utf-8"))
        ]
