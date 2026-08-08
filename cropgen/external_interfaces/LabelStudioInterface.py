from label_studio_sdk import Client
import json
import os
from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.simplify_export import (
    simplify_and_save,
    load_simplified_export,
)
from cropgen.shared.LSTypedDicts.aggregates import LabelStudioTask
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedTask,
    SimplifiedAnnotation,
)
from pathlib import Path
from cropgen.processing.AnnotatedPage import AnnotatedPage
from PIL import Image, ImageOps


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
        "server_url",
        "token" "project_id",
    )

    def __init__(
        self,
        paths: PathBundle,
        server_url: str,
        token: str,
        project_id: int = 4,
        online: bool = True,
    ):
        """
        Inicializa la interfaz de Label Studio a partir de un PathBundle.
        Carga los exports locales (raw y simplified) y la lista de usuarios si existen.
        """
        self.project = None
        self.paths = paths
        self.raw_export_filepath = paths.raw_export_filepath
        self.simplified_export_filepath = paths.simplified_filepath
        self.online = online
        self.project_id = project_id

        exists_raw = paths.raw_export_filepath.exists()
        exists_sim = paths.simplified_filepath.exists()

        if not online:

            if not exists_sim and not exists_raw:
                print(
                    f"No existe export local crudo ni simplificado, y se ha seleccionado online = False."
                )

            if not exists_raw and exists_sim:
                print(
                    f"No existe export crudo local en {paths.raw_export_filepath}, y online = False,"
                    f"empleando directamente el simplificado local en {paths.simplified_filepath}."
                )

            else:
                simplify_and_save(paths.raw_export_filepath, paths.simplified_filepath)

            return

        self.token = token
        self.url = server_url

        self.fetch_and_simplify()

        if paths.usernames_filepath.exists():
            self.usernames: list[str] = list(
                json.loads(paths.usernames_filepath.read_text(encoding="utf-8"))
            )
        else:
            self.usernames: list[str] = []

    @classmethod
    def from_env(
        cls,
        paths: PathBundle,
        online: bool = True,
        project_id: int = 4,
        token_env_var: str = "LS_TOKEN",
        url_env_var: str = "LS_URL",
    ) -> "LabelStudioInterface":
        if token_env_var not in os.environ:
            raise ValueError(
                f"{token_env_var} no está presente en las variables de entorno."
            )
        elif url_env_var not in os.environ:
            raise ValueError(
                f"{url_env_var} no está presente en las variables de entorno."
            )

        token = str(os.getenv(token_env_var))
        url = str(os.getenv(url_env_var))

        obj = cls(paths, url, token, project_id, online)

        obj.usernames = json.loads(
            obj.paths.usernames_filepath.read_text(encoding="utf-8")
        )
        return obj

    def __repr__(self):
        return f"<LabelStudioInterface con REF = {self.raw_export_filepath} y SEF={self.simplified_export_filepath}.>"

    @staticmethod
    def _get_latest_update_of_project(project) -> str:
        """
        Devuelve la fecha de la última actualización de una tarea en el proyecto de Label Studio.
        """
        updated_at = project.get_paginated_tasks(
            ordering=["-updated_at"], page=1, page_size=1
        )["tasks"][0]["updated_at"]
        return str(updated_at)

    def fetch_and_simplify(
        self,
        force_update: bool = False,
    ) -> bool:
        """
        Actualiza los archivos de exportación desde Label Studio si hay cambios o si se fuerza la actualización.
        Descarga los datos, los guarda y regenera el export simplificado.
        Devuelve True si se ha actualizado, False si ya estaba actualizado.
        """
        if not self.online:
            print(
                f"LSI configurado con online={self.online}, por tanto no se actualiza."
            )
            self.usernames = json.loads(
                self.paths.usernames_filepath.read_text(encoding="utf-8")
            )
            return False

        ls_client = Client(url=self.url, api_key=self.token)
        project = ls_client.get_project(id=self.project_id)

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
        self.paths.usernames_filepath.write_text(
            json.dumps(ordered_usernames), encoding="utf-8"
        )

        # comprobamos si hace falta actualizar
        latest_update = LabelStudioInterface._get_latest_update_of_project(project)

        if self.paths.raw_export_filepath.exists() and not force_update:
            loaded_export = self.raw_tasks
            if loaded_export:
                local_last_update = max(task.updated_at for task in loaded_export)
                if (
                    latest_update <= local_last_update
                ) and self.paths.simplified_filepath.exists():
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
        self.paths.raw_export_filepath.write_text(
            json.dumps(dump_data), encoding="utf-8"
        )

        simplify_and_save(
            self.paths.raw_export_filepath, self.paths.simplified_filepath
        )
        return True

    @property
    def raw_tasks(self) -> list[LabelStudioTask]:
        """
        Devuelve la lista de tareas raw descargadas de Label Studio
        """
        raw_tasks = [
            LabelStudioTask.model_validate(task_dict)
            for task_dict in json.loads(
                self.raw_export_filepath.read_text(encoding="utf-8")
            )
        ]

        raw_tasks.sort(key=lambda task: task.id)

        return raw_tasks

    @property
    def simplified_tasks(self) -> list[SimplifiedTask]:
        """
        Devuelve la lista de tareas simplificadas descargadas de Label Studio
        """
        simplified_tasks = [
            SimplifiedTask.model_validate(task_dict)
            for task_dict in json.loads(
                self.simplified_export_filepath.read_text(encoding="utf-8")
            )
        ]

        simplified_tasks.sort(key=lambda task: task.id)

        return simplified_tasks

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
        for tsk in self.simplified_tasks:
            if int(tsk.id) > index:
                return items
            elif tsk.id == index:
                items.extend(tsk.annotations)
        return items

    def get_annotated_page(self, index: int, subindex: int = 0) -> AnnotatedPage:
        task = [task for task in self.simplified_tasks if int(task.id) == index][0]

        if subindex >= len(task.annotations):
            raise ValueError(
                f"No enough annotations on this task: {len(task.annotations)=} <= {subindex=}"
            )

        annotation = task.annotations[subindex]

        img_path = self.paths.get_image_path_from_task(task)

        if img_path is None:
            raise ValueError(f"No hay imagen para la tarea {task.id}")

        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            raise ValueError(f"Error cargando {img_path}: {e}")

        return AnnotatedPage(
            annotation,
            img,
            unrotate=False,
            usernames_labelstudio=self.usernames,
            process_images=True,
        )

    @property
    def annotated_pages(self) -> list[AnnotatedPage]:

        pages: list[AnnotatedPage] = []
        for task in self.simplified_tasks:
            img_path = self.paths.get_image_path_from_task(task)

            if img_path is None:
                print(f"No hay imagen para la tarea {task.id}")
                continue

            try:
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
                continue

            annotations = [
                AnnotatedPage(
                    ann,
                    img,
                    unrotate=False,
                    usernames_labelstudio=self.usernames,
                    process_images=True,
                )
                for ann in task.annotations
            ]

            if len(annotations) > 1:
                pages.append(AnnotatedPage.combine_annotations(*annotations))
            elif len(annotations) == 1:
                pages.append(annotations[0])
            else:
                print(f"Aviso: La tarea {task.id} no tiene anotaciones.")

        return pages
