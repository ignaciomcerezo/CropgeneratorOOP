from typing import Annotated
from cropgen.shared.LSTypedDicts.results import ImageBaseResult
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
from cropgen.processing.annotated_page import AnnotatedPage
from PIL import Image, ImageOps
import numpy as np
from urllib.parse import unquote
from tqdm.auto import tqdm


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
        self._raw_tasks_cache: list[LabelStudioTask] | None = None
        self._simplified_tasks_cache: list[SimplifiedTask] | None = None
        self._annotated_pages_cache: list[AnnotatedPage] | None = None
        self._task_image_path_cache: dict[int, Path | None] = {}

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
        ls_token: str | None = None,
        ls_server_url: str | None = None,
        token_env_var: str = "LS_TOKEN",
        url_env_var: str = "LS_URL",
    ) -> "LabelStudioInterface":

        if (token_env_var not in os.environ) and (ls_token is None):
            raise ValueError(
                f"{token_env_var} no está presente en las variables de entorno."
            )
        elif (url_env_var not in os.environ) and (ls_server_url is None):
            raise ValueError(
                f"{url_env_var} no está presente en las variables de entorno."
            )

        token = ls_token if ls_token is not None else str(os.getenv(token_env_var))
        url = (
            ls_server_url if ls_server_url is not None else str(os.getenv(url_env_var))
        )

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
        self._invalidate_caches()
        return True

    def _invalidate_caches(self) -> None:
        self._raw_tasks_cache = None
        self._simplified_tasks_cache = None
        self._annotated_pages_cache = None
        self._task_image_path_cache = {}

    @property
    def raw_tasks(self) -> list[LabelStudioTask]:
        """
        Devuelve la lista de tareas raw descargadas de Label Studio
        """
        if self._raw_tasks_cache is None:
            raw_tasks = [
                LabelStudioTask.model_validate(task_dict)
                for task_dict in json.loads(
                    self.raw_export_filepath.read_text(encoding="utf-8")
                )
            ]

            raw_tasks.sort(key=lambda task: task.id)
            self._raw_tasks_cache = raw_tasks

        return self._raw_tasks_cache

    @property
    def simplified_tasks(self) -> list[SimplifiedTask]:
        """
        Devuelve la lista de tareas simplificadas descargadas de Label Studio
        """
        if self._simplified_tasks_cache is None:
            simplified_tasks = [
                SimplifiedTask.model_validate(task_dict)
                for task_dict in json.loads(
                    self.simplified_export_filepath.read_text(encoding="utf-8")
                )
            ]

            simplified_tasks.sort(key=lambda task: task.id)
            self._simplified_tasks_cache = simplified_tasks

        return self._simplified_tasks_cache

    def _get_image_path_from_task_cached(self, task: SimplifiedTask) -> Path | None:
        task_id = int(task.id)
        if task_id not in self._task_image_path_cache:
            self._task_image_path_cache[task_id] = self.paths.get_image_path_from_task(
                task
            )
        return self._task_image_path_cache[task_id]

    def _load_task_image(self, task: SimplifiedTask) -> Image.Image:
        img_path = self._get_image_path_from_task_cached(task)

        if img_path is None:
            raise ValueError(f"No hay imagen para la tarea {task.id}")

        try:
            with Image.open(img_path) as img:
                return ImageOps.exif_transpose(img).copy()
        except Exception as e:
            raise ValueError(f"Error cargando {img_path}: {e}")

    def _build_annotated_page_from_task(
        self,
        task: SimplifiedTask,
        *,
        subindex: int | None = None,
        process_images: bool = True,
    ) -> AnnotatedPage:
        if subindex is not None and subindex >= len(task.annotations):
            raise ValueError(
                f"No enough annotations on task of task_id={task.id}: {len(task.annotations)=} <= {subindex=}."
            )

        valid_annotations = (
            [task.annotations[subindex]] if subindex is not None else task.annotations
        )

        if len(valid_annotations) == 0:
            raise ValueError(f"Aviso: La tarea {task.id} no tiene anotaciones.")

        img = self._load_task_image(task)
        page_name = self._get_page_from_task(task)

        if len(valid_annotations) == 1:
            return AnnotatedPage(
                valid_annotations[0],
                img,
                usernames_labelstudio=self.usernames,
                process_images=process_images,
                page=page_name,
            )

        if not process_images:
            return AnnotatedPage.combine_annotations(
                *[
                    AnnotatedPage(
                        ann,
                        img,
                        usernames_labelstudio=self.usernames,
                        process_images=False,
                        page=page_name,
                    )
                    for ann in valid_annotations
                ]
            )

        # Reuse stroke/background from the first annotation to avoid re-running
        # expensive image separation for every annotation in the same task.
        first = AnnotatedPage(
            valid_annotations[0],
            img,
            usernames_labelstudio=self.usernames,
            process_images=True,
            page=page_name,
        )

        others = [
            AnnotatedPage(
                ann,
                img,
                usernames_labelstudio=self.usernames,
                page=page_name,
                stroke=first.stroke,
                background=first.background,
            )
            for ann in valid_annotations[1:]
        ]

        return AnnotatedPage.combine_annotations(first, *others)

    def get_annotated_pages(
        self,
        *,
        process_images: bool = True,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> list[AnnotatedPage]:
        if process_images and use_cache and self._annotated_pages_cache is not None:
            return self._annotated_pages_cache

        pages: list[AnnotatedPage] = []
        iterator = (
            tqdm(self.simplified_tasks) if show_progress else self.simplified_tasks
        )

        for task in iterator:
            try:
                pages.append(
                    self._build_annotated_page_from_task(
                        task, process_images=process_images
                    )
                )
            except ValueError as e:
                print(e)

        if process_images and use_cache:
            self._annotated_pages_cache = pages

        return pages

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

    def get_annotated_page(
        self,
        *,
        task_id: int | None = None,
        page: str | None = None,
        subindex: int | None = None,
        process_images: bool = True,
    ) -> AnnotatedPage:
        """
        Returns the annotated page instance corresponding to the index/page and the subindex specified.
        Subindex is the index of the annotation in the task corresponding to the index/page specified.
        """
        if (task_id is not None) and (page is not None):
            raise ValueError(
                f"Only of of task_id and page must be specified, but got {task_id=} and {page=}"
            )
        elif task_id is not None:
            possible_tasks = [
                task for task in self.simplified_tasks if int(task.id) == task_id
            ]
        else:
            possible_tasks = [
                task
                for task in self.simplified_tasks
                if self._get_page_from_task(task) == page
            ]

        specifier_str = f"{task_id=}" if task_id is not None else f"{page=}"

        if len(possible_tasks) == 0:
            raise IndexError(
                f"There is no task verifying {specifier_str}. Make sure the index is correct, if given a task_id, or the page format is correct, if given a page."
            )
        elif len(possible_tasks) != 1:
            raise IndexError(
                f"There are too many ({len(possible_tasks)}) tasks verifying the given condition {specifier_str}."
            )

        task = possible_tasks[0]

        return self._build_annotated_page_from_task(
            task, subindex=subindex, process_images=process_images
        )

    @staticmethod
    def _get_page_from_task(task: SimplifiedTask | LabelStudioTask) -> str:
        return Path(unquote(task.data.image_url)).stem

    def page_names(self):
        return tuple([self._get_page_from_task(task) for task in self.simplified_tasks])
