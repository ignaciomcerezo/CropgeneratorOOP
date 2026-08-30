from cropgen.external_interfaces.external_interface import ExternalInterface
import urllib
from shapely.geometry import Polygon
from cropgen.shared.image_processing import separate_background_and_stroke
from typing import Annotated
from cropgen.external_interfaces.label_studio.ls_typed_dicts import (
    ImageBaseResult,
    RectangleResult,
    LabelStudioTask,
    SimplifiedTask,
    SimplifiedAnnotation,
)
from label_studio_sdk import Client
import json
import os
from cropgen.shared.path_bundle import PathBundle
from cropgen.external_interfaces.label_studio.helpers.simplify_export import (
    simplify_and_save,
    load_simplified_export,
)
from cropgen.external_interfaces.label_studio.helpers.json_conversor import (
    pair_lines,
    extract_bounds,
    calculate_reading_angle,
)
from pathlib import Path
from cropgen.processing.annotated_page import AnnotatedPage
from PIL import Image, ImageOps
import numpy as np
from tqdm.auto import tqdm
from urllib.parse import unquote as url_unquote


class _UnknownUsernames:
    def __init__(self):
        pass

    def __getitem__(self, index):
        return "Offline-Unknown"

    def __len__(self):
        return 2**20


class LabelStudioInterface(ExternalInterface):
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
        Mantiene la instancia ligera; la descarga y actualización se ejecutan en setup().
        """
        self.project = None
        self.paths = paths
        self.raw_export_filepath = paths.raw_export_filepath
        self.simplified_export_filepath = paths.simplified_filepath
        self.online = online
        self.project_id = project_id
        self.token = token
        self.url = server_url
        self._raw_tasks_cache: list[LabelStudioTask] | None = None
        self._simplified_tasks_cache: list[SimplifiedTask] | None = None
        self._annotated_pages_cache: list[AnnotatedPage] | None = None
        self._task_image_path_cache: dict[int, Path | None] = {}
        self.usernames: list[str] | _UnknownUsernames = []

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
        ordered_usernames: list[str] = []
        if user_ids:
            for x in range(max(user_ids) + 1):
                if x in user_ids:
                    ordered_usernames.append(
                        [u.username for u in users if u.id == x][0]
                    )
                else:
                    ordered_usernames.append("Impossible LS user")

        self.usernames = ordered_usernames

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

    def _load_raw_task_image(self, task: SimplifiedTask) -> Image.Image:
        img_path = self.get_raw_image_path_from_task(task)

        if img_path is None:
            raise ValueError(f"No hay imagen para la tarea {task.id}")

        try:
            with Image.open(img_path) as img:
                return ImageOps.exif_transpose(img).copy()
        except Exception as e:
            raise ValueError(f"Error cargando {img_path}: {e}")

    def _get_completer(self, annotation: SimplifiedAnnotation):
        completer_index = annotation.completed_by
        return (
            self.usernames[completer_index]
            if completer_index < len(self.usernames)
            else "Impossible User"
        )

    def _get_updater(self, annotation: SimplifiedAnnotation):
        updater_index = annotation.completed_by
        return (
            self.usernames[updater_index]
            if updater_index < len(self.usernames)
            else "Impossible User"
        )

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

    @staticmethod
    def _get_page_from_task(task: SimplifiedTask | LabelStudioTask) -> str:
        return Path(url_unquote(task.data.image_url)).stem

    def page_names(self):
        return tuple([self._get_page_from_task(task) for task in self.simplified_tasks])

    @staticmethod
    def get_image_stem_from_task(
        task: dict | LabelStudioTask | SimplifiedTask,
    ) -> str:
        task: LabelStudioTask | SimplifiedTask = (
            LabelStudioInterface._simplified_or_raw(task)
        )
        clean_url = url_unquote(task.data.image_url)
        filename = Path(clean_url.split("?")[0].split("/")[-1])
        return filename.stem

    def get_raw_image_path_from_task(
        self,
        task: LabelStudioTask | SimplifiedTask,
    ) -> Path | None:
        """
        Returns the local path to the corresponding raw image.
        If it cant find it, returns None.
        """
        stem = self.get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the raw image for task: ", task.id)

        filepath = self.paths.get_raw_image_path(stem)

        if filepath.exists():
            return filepath
        print("Could not find the raw image for task: ", task.id)
        return None

    def get_stroke_image_path_from_task(
        self, task: LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Returns the local path to the corresponding stroke image.
        If it cant find it, returns None.
        """
        stem = self.get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the stroke image for task: ", task.id)

        filepath = self.paths.get_stroke_image_path(stem)

        if filepath.exists():
            return filepath
        return None

    def get_background_image_path_from_task(
        self, task: LabelStudioTask | SimplifiedTask
    ) -> Path | None:
        """
        Returns the local path to the corresponding background image.
        If it cant find it, returns None.
        """
        stem = self.get_image_stem_from_task(task)
        if stem is None:
            raise ValueError("Could not find the background image for task: ", task.id)

        filepath = self.paths.get_background_image_path(stem)

        if filepath.exists():
            return filepath
        return None

    @staticmethod
    def _simplified_or_raw(
        obj: dict | SimplifiedTask | LabelStudioTask,
    ) -> SimplifiedTask | LabelStudioTask:

        if isinstance(obj, dict):
            try:
                converted_obj = SimplifiedTask.model_validate(obj)
                return converted_obj
            except:
                try:
                    converted_obj = LabelStudioTask.model_validate(obj)
                    return converted_obj
                except:
                    raise TypeError(
                        "Se ha pasado un objeto que no cumple ninguna de las dos."
                    )
        elif isinstance(obj, (SimplifiedTask, LabelStudioTask)):
            return obj
        else:
            raise TypeError("Se ha pasado un tipo incorrecto")

    def parts_managed(self):
        return [
            "metadata",
            "rotations",
        ]

    def parts_required(self):
        return []

    def setup(self) -> None:
        """
        Downloads or refreshes the Label Studio exports when online and generates the
        polygon, metadata and transcription JSON files from the annotations.
        """
        if self.online:
            self.fetch_and_simplify()
        else:
            exists_raw = self.paths.raw_export_filepath.exists()
            exists_sim = self.paths.simplified_filepath.exists()

            if not exists_sim and not exists_raw:
                print(
                    f"No existe export local crudo ni simplificado, y se ha seleccionado online = False."
                )

            if not exists_raw and exists_sim:
                print(
                    f"No existe export crudo local en {self.paths.raw_export_filepath}, y online = False,"
                    f"empleando directamente el simplificado local en {self.paths.simplified_filepath}."
                )
            else:
                simplify_and_save(
                    self.paths.raw_export_filepath, self.paths.simplified_filepath
                )

            if self.paths.usernames_filepath.exists():
                self.usernames = list(
                    json.loads(
                        self.paths.usernames_filepath.read_text(encoding="utf-8")
                    )
                )
            else:
                self.usernames = _UnknownUsernames()

        for task in self.simplified_tasks:
            image_url = task.data.image_url
            task_id = task.id
            page = Path(url_unquote(image_url)).stem
            for subindex, simplified_ann in enumerate(task.annotations):

                transcriptions: list[str] = []
                poly_coords: list[list[tuple[float, float]]] = []
                ids: list[str] = []
                rotations: list[float] = []
                completer: str = self._get_completer(simplified_ann)
                updater: str = self._get_updater(simplified_ann)
                ann_id = simplified_ann.id

                results = simplified_ann.result
                box2text, id2boxres, id2txtres = pair_lines(results)

                trios = [
                    (
                        id2boxres[key],
                        id2txtres[box2text[key]],
                        key + "-" + box2text[key],
                    )
                    for key in id2boxres
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

                (self.paths.transcription_path / json_name).write_text(
                    json.dumps(transcriptions)
                )
                (self.paths.polygons_path / json_name).write_text(
                    json.dumps(poly_coords)
                )
                (self.paths.rotations_path / json_name).write_text(
                    json.dumps(rotations)
                )
                (self.paths.ids_path / json_name).write_text(json.dumps(ids))
                metadata = {
                    "page": page,
                    "task_id": task_id,
                    "completer": completer,
                    "updater": updater,
                    "subindex": subindex,
                    "ann_id": ann_id,
                    "order": len(trios),
                    "image_path": str(self.paths.raw_images_path / f"{page}.png"),
                    "ids_path": str(self.paths.ids_path / json_name),
                    "txt_path": str(self.paths.transcription_path / json_name),
                    "polygons_path": str(self.paths.polygons_path / json_name),
                    "rotations_path": str(self.paths.rotations_path / json_name),
                    "polygons_are_in_percentage": True,
                    "source": "Label Studio",
                }
                (self.paths.metadata_path / json_name).write_text(json.dumps(metadata))
