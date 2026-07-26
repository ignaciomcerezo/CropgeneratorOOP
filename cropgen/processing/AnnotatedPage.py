from typing import Literal
import re
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from cropgen.processing.ImageBox import ImageBox
from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.TextFragment import TextFragment
from cropgen.processing.helpers.PairingErrors import (
    NoAssociationError,
    MultipleAssociationError,
    RepeatedSameAssociationError,
    SameToSameAssociation,
)
from cropgen.processing.helpers.helper_to_classes import (
    get_connected_components,
    get_dominant_color,
    compose_collage,
    subdictionary,
)
from cropgen.processing.helpers.text_regularization import regularize_text
from cropgen.shared.LSTypedDicts.results import (
    RectangleResult,
    PolygonResult,
    RelationResult,
    TextRegionResult,
)
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedAnnotation,
    SimplifiedResultItem,
    SimplifiedTextCorrectionResult,
)
from cropgen.shared.default_parameters import (
    big_box_threshold,
    min_nodes_for_big_box_removal,
)
from cropgen.shared.display import display


class AnnotatedPage:
    """
    Clase que representa una única anotación. Recoge la información sobre las cajas-imagen y los fragmentos
    de texto (con sus relaciones), construye el grafo de adyacencia, crea los párrafos y ordena la información
    y la hace accesible de forma que la función augment_data tenga menor complejidad.
    """

    n_annotation_errors = 0
    warn_unrotate = True
    warn_process_images = True
    min_nodes_for_big_box_removal = min_nodes_for_big_box_removal
    __slots__ = (
        "background_color",
        "image_boxes",
        "text_fragments",
        "task_id",
        "__graph",
        "last_update_time",
        "completer",
        "updater",
        "paragraphs",
        "annotation_unique_id",
        "process_images",
        "line_separator",
    )

    def __init__(
        self,
        ann: SimplifiedAnnotation,
        img: Image.Image,
        unrotate: bool = False,
        usernames_labelstudio: list[str] | None = None,
        line_separtor: str = "\n",
        process_images: bool = True,
    ):

        if unrotate and AnnotatedPage.warn_unrotate:
            print(
                "[!!!] Usar unrotate = True destruye la información sobre la posición del crop en "
                "la instancia de AnnotatedPage. Además, reduce la calidad de las imágenes por usar "
                "interpolación bicúbica, y esta misma interpolación introduce artefactos visuales "
                "en los bordes de la imagen. También invalida la forma en la que se generan los párrafos, "
                "la transcripción global (y la de los clusters) y los starting_indices.\n"
                "Úsese solamente en caso de revisión manual de las imágenes, y NO para el código de "
                "generación del dataset."
            )
            AnnotatedPage.warn_unrotate = False

        if not (process_images) and AnnotatedPage.warn_process_images:
            print(
                "[!!!] Usar fake_images = True evita usar las imágenes y produce recortes vacíos.\n"
                "Úsese solo en caso de testeo y NO para el código de generación del dataset."
            )
            AnnotatedPage.warn_process_images = False
        assert (
            usernames_labelstudio is not None
        ), "Es necesario proporcionar la lista de usernames de LS para generar la anotación."

        # corrige los resultados realizando las sustituciones
        self.task_id = int(ann.task)
        results: list[SimplifiedResultItem] = ann.result

        self.background_color = get_dominant_color(img)

        img_results_list: list[RectangleResult | PolygonResult] = [
            r for r in results if isinstance(r, (RectangleResult, PolygonResult))
        ]

        txt_results_list: list[SimplifiedTextCorrectionResult] = [
            r for r in results if isinstance(r, SimplifiedTextCorrectionResult)
        ]

        self.image_boxes: dict[str, ImageBox] = (
            {  # conjunto de cajas-imagen (instancias de ImageBox)
                img_result.id: ImageBox.from_image_result(
                    img_result, self.task_id, img, unrotate
                )
                for img_result in img_results_list
            }
        )

        self.text_fragments: dict[str, TextFragment] = {
            txt_result.id: TextFragment(
                id=txt_result.id,
                text=" ".join(txt_result.value.text).strip(),
                task_id=self.task_id,
            )
            for txt_result in txt_results_list
        }

        self.process_images = process_images
        self.line_separator = line_separtor

        self._setup_mappings(
            results
        )  # guardamos en cada dataclass los otros objetos que tiene asociados mediante una relación de external_interfaces

        self.assert_pairing()  # nos aseguramos de que todas las imágenes tengan fragmento, y viceversa

        self.__graph: dict[str, set[str]] = (
            self._build_intersection_graph()
        )  # construimos el grafo de intersecciones entre cajas-imagen

        # colocamos las componentes conexas siguiendo el orden de lectura.

        connected_components = get_connected_components(self.__graph)

        box_ccs = [
            [self.image_boxes[box_id] for box_id in component]
            for component in connected_components
        ]

        subgraphs_ccs = [
            subdictionary(component, subdictionary(component, self.graph))
            for component in connected_components
        ]

        # generamos los párrafos (componentes conexas con información extra), que añaden automáticamente información sobre
        # los centroides corregidos a cada caja-imagen. El sorted se ejecuta atuomáticamente, y se hace usando el orden
        # naif.
        self.paragraphs: list[Paragraph] = sorted(
            [
                Paragraph(box_cc, task_id=self.task_id, subgraph=subgraph)
                for (box_cc, subgraph) in zip(box_ccs, subgraphs_ccs)
            ]
        )

        # notemos que solamente las imágenes que estén en un párrafo tienen sindex...
        self._correct_text_and_set_sindices()

        self.last_update_time = " ".join(
            ann.updated_at.replace("Z", "").split("T")
        )  # última actualización de la tarea

        completer_index = ann.completed_by
        updater_index = ann.updated_by
        self.completer = (
            usernames_labelstudio[completer_index]
            if completer_index < len(usernames_labelstudio)
            else "Unknown"
        )
        self.updater = (
            usernames_labelstudio[updater_index]
            if updater_index < len(usernames_labelstudio)
            else "Unknown"
        )
        self.annotation_unique_id = ann.id

    @property
    def order(self) -> int:
        """Número total de imágenes (si no hay PairingError también será el total de fragmentos)"""
        return len(self.graph)

    @property
    def graph(self) -> dict[str, set[str]]:
        """Grafo de adyacencia de las ImageBox.id dado por las intersecciones de sus crops."""
        return self.__graph

    @graph.setter
    def graph(self, value):
        raise ValueError(
            "Por causas de starting_index y composición del documento, no es posible modificar el grafo!"
        )

    def _setup_mappings(self, results: list[SimplifiedResultItem]) -> None:
        """
        A partir de las relaciones creadas en cada tarea de LS, genera respectivos diccionarios:
            1. img2text_rel: dict[str, str], box_id -> fragment_id,
        que lleva el ID de una caja-imagen al id de un fragmento, y
            2. text2img_rel: dict[str, str], fragment_id -> box_id,
        que lleva el ID de un fragmento a su caja-imagen correspondiente.
        """

        for r in results:
            if isinstance(r, RelationResult):  # si el resultado es una relación
                source_id, target_id = r.from_id, r.to_id

                if (source_id in self.image_boxes) and (
                    target_id in self.text_fragments
                ):
                    # asociación caja-imagen -> fragmento
                    box_id, fragment_id = source_id, target_id
                elif (source_id in self.text_fragments) and (
                    target_id in self.image_boxes
                ):
                    # asociación fragmento -> caja-imagen
                    box_id, fragment_id = target_id, source_id
                elif (source_id in self.image_boxes) and (
                    target_id in self.image_boxes
                ):
                    AnnotatedPage._register_error()
                    # asociación caja-imagen -> caja-imagen (error de anotación)
                    print(f"(Task {self.task_id}) Asociación caja-imagen->caja-imagen:")
                    print("Caja 1 (source):")
                    display(self.image_boxes[source_id].crop)
                    print("Caja 2 (target):")
                    display(self.image_boxes[target_id].crop)
                    continue
                elif (source_id in self.text_fragments) and (
                    target_id in self.text_fragments
                ):
                    AnnotatedPage._register_error()
                    # asociación fragmento -> fragmento (error de anotación)
                    print(f"(Task {self.task_id}) Asociación texto->texto.")
                    print(self.text_fragments[source_id].text)
                    print(self.text_fragments[target_id].text)
                    continue
                else:
                    AnnotatedPage._register_error()
                    # otro tipo de asociación (extraña)
                    print(f"(Task {self.task_id}) Asociación rara.")
                    continue

                # comprobamos ahora que de cada objeto sale o entra una única relación, ni más ni menos.
                # (es decir, que un fragmento solamente está conectado a una imagen y solo una vez, y viceversa)

                image_box = self.image_boxes[box_id]
                text_fragment = self.text_fragments[fragment_id]

                image_box.associate_fragment(text_fragment)
                text_fragment.associate_box(image_box)

    def _correct_text_and_set_sindices(self):
        sindex = 0  # índices de inicio de cada fragmento de texto
        # notemos que solamente las imágenes que estén en un párrafo tienen sindex...
        for paragraph_index, paragraph in enumerate(self.paragraphs):
            paragraph.index = paragraph_index
            separator = "@SEP@"
            raw_separated_transcription = paragraph.transcription(separator)
            regularized_transcriptions = regularize_text(
                raw_separated_transcription
            ).split(separator)

            assert len(regularized_transcriptions) == len(
                paragraph.text_fragments
            ), "El número de transcripciones regularizadas y el número de líneas de texto no coinciden"

            for fragment, new_transcription in zip(
                paragraph.text_fragments, regularized_transcriptions
            ):
                fragment.text = new_transcription
                fragment.starting_index = sindex
                sindex += len(fragment.text) + len(self.line_separator)

    def assert_pairing(self):
        """
        Compruba que todas las cajas están asociadas a un único texto, y viceversa
        """
        for fragment in self.text_fragments.values():
            if any(
                [isinstance(obj, TextFragment) for obj in fragment.associated_boxes]
            ):
                raise SameToSameAssociation(fragment)

            if len(set(fragment.associated_boxes)) != len(fragment.associated_boxes):
                raise RepeatedSameAssociationError(fragment)
            elif len(fragment.associated_boxes) > 1:
                raise MultipleAssociationError(fragment)
            elif len(fragment.associated_boxes) == 0:
                raise NoAssociationError(fragment)

        for box in self.image_boxes.values():
            if any([isinstance(obj, ImageBox) for obj in box.associated_fragments]):
                raise SameToSameAssociation(box)

            if len(set(box.associated_fragments)) != len(box.associated_fragments):
                raise RepeatedSameAssociationError(box)
            elif len(box.associated_fragments) > 1:
                raise MultipleAssociationError(box)
            elif len(box.associated_fragments) == 0:
                raise NoAssociationError(box)

    def __repr__(self):
        return f"<Annotation of task {self.task_id} of order {self.order}. Completed by {self.completer}, last updated by {self.updater} at {self.last_update_time}>"

    def _build_intersection_graph(self):
        """
        Genera el grafo de intersecciones de una anotación.
        Devuelve un diccionario de adyacencia {box_id: set(id_adyacentes)}.
        """
        adj = {image_box_id: set() for image_box_id in self.image_boxes}
        for i, box1 in enumerate(self.image_boxes.values()):
            for j, box2 in enumerate(self.image_boxes.values()):
                if j <= i:
                    continue

                if box1.polygon.intersects(box2.polygon):
                    adj[box1.id].add(box2.id)
                    adj[box2.id].add(box1.id)

        return adj

    def generate_collage(
        self,
        box_id_sequence: set[str] | list[str],
        background_color: (
            tuple[int, int, int] | tuple[int, int, int, int] | None
        ) = None,
    ) -> Image.Image:
        """
        Genera el collage de recortes para una secuencia de ids de cajas (un subgrafo), colocando en sus posiciones en
        la página original los recortes, rellenando el resto con el color promedio de la imagen y recortando la imagen
        al tamaño mínimo que contiene todos los recortes colocados.
        """
        if not self.process_images:
            return Image.Image()

        if not isinstance(box_id_sequence, set):
            if len(box_id_sequence) != len(set(box_id_sequence)):
                raise ValueError("Hay cajas-imagen repetidas en generate_collage()")
            box_id_sequence = set(box_id_sequence)

        subgraph_image_boxes = [self.image_boxes[box_id] for box_id in box_id_sequence]

        background_color = (
            background_color if background_color is not None else self.background_color
        )
        return compose_collage(
            subgraph_image_boxes,
            (
                background_color
                if not isinstance(background_color, tuple)
                else self.background_color
            ),
        )

    # def trim_star_nodes(
    #     self,
    #     relative_threshold: float = big_box_threshold,
    # ) -> None:
    #     """
    #     Elimina los nodos con una conectividad mayor a relative_threshold
    #     """

    #     adj_graph = self.__graph.copy()
    #     nodes_to_remove = []

    #     for rid, neighbors in adj_graph.items():
    #         if (
    #             len(neighbors) / (len(adj_graph) - 1) > relative_threshold
    #         ):  # si pasa el umbral
    #             nodes_to_remove.append(rid)

    #     if len(nodes_to_remove) > 0:
    #         print(
    #             f"Eliminando {len(nodes_to_remove)} nodos estrellados de la tarea {self.task_id}."
    #         )

    #     for rid in nodes_to_remove:
    #         self.image_boxes[rid].fragment.starting_index = -1

    #         del adj_graph[rid]  # quitamos el nodo en forma de estrella

    #         for other in adj_graph:  # eliminamos todas sus referencias
    #             adj_graph[other].discard(rid)

    #     self.__graph = adj_graph

    def cluster_reading_order(
        self,
        box_ids: list["str"],
    ) -> tuple[Image.Image, str, int]:
        """
        Dada una lista de IDs de cajas-imagen, devuelve:
        - su collage correspondiente
        - la transcripción en el orden de lectura
        - el índice de inicio de este bloque en la transcripción total.
        """

        collage = self.generate_collage(box_ids)

        fragments = [self.image_boxes[box_id].fragment for box_id in box_ids]
        # usando .starting_index estamos usando el mismo orden de lectura de image_boxes
        fragments: list[TextFragment] = sorted(
            fragments, key=lambda x: x.starting_index
        )  # ty:ignore[no-matching-overload]

        transcription = self.line_separator.join(
            [fragment.text for fragment in fragments]
        )
        if not fragments:
            raise ValueError(
                f"No se puede llamar cluster_reading_order si no hay fragmentos asociados ({self.task_id}) -> {box_ids=}"
            )

        if fragments[0].starting_index is None:
            raise ValueError(
                "Los fragmentos dados no tienen índices de inicio asignados."
            )

        sindex = int(fragments[0].starting_index)

        return collage, transcription, sindex

    def are_in_same_cc(self, box_id_sequence: list[str]) -> bool:
        """
        Devuelve si una secuencia de ids de cajas está o no en la misma componente conexa
        """
        if not box_id_sequence:
            return True

        first_box_id = box_id_sequence[0]

        for paragraph in self.paragraphs:
            if first_box_id in paragraph.image_boxes_ids:
                break
        else:
            raise ValueError(
                "El primer box_id no pertenece a ninguna componente conexa (párrafo) de esta anotación. ¿Pertenece a esta página?"
            )

        return set(box_id_sequence[1:]).issubset(set(paragraph.image_boxes_ids))

    @property
    def n_paragraphs(self):
        """Número de párrafos de la anotación"""
        return len([paragraph for paragraph in self.paragraphs if (len(paragraph) > 1)])

    @property
    def is_single_paragraph(self):
        return self.n_paragraphs == 1

    @staticmethod
    def _register_error():
        AnnotatedPage.n_annotation_errors += 1

    def fragments_without_paragraph(self) -> list[TextFragment]:
        """
        Devuelve una lista de fragmentos sin párrafo. Estos pueden venir de dos fuentes:
            1. Son fragmentos aislados del resto durante las anotaciones.
        Anteriormente podían ser fragmentos de conectividad alta desconectados usando trim_star_nodes,
        pero, a partir del cambio de paradigma hacia lecturas puramente lineales (grafos de los párrafos
        de tipo P_k) se eliminó esta dinámica.
        """
        in_paragraph = []
        out_paragraph = []
        for paragraph in self.paragraphs:
            in_paragraph += [f.id for f in paragraph.text_fragments]
        if len(in_paragraph) == len(self.text_fragments):
            return []

        for f in self.text_fragments.values():
            if f.id not in in_paragraph:
                out_paragraph.append(f.id)
        return [self.text_fragments[fragment_id] for fragment_id in out_paragraph]

    def get_average_rotation(self, img_box_ids: Iterable[str]) -> float:
        """
        Devuelve la rotación promedio de un grupo de cajas-imagen dados sus ids.
        """
        image_boxes = [self.image_boxes[box_id] for box_id in img_box_ids]
        areas = [box.polygon.area for box in image_boxes]
        angles_in_radians = [np.radians(box.rotation) for box in image_boxes]

        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))

        return -np.degrees(np.arctan2(sum_sin, sum_cos))

    def represent_by_ids(
        self,
        image_box_ids: list[str],
        represent_polygon: bool = True,
        represent_mbr: bool = False,
        polygon_color: tuple[int, int, int] | tuple[int, int, int, int] = (255, 0, 0),
        mbr_color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 255, 0),
        background_color: (
            tuple[int, int, int] | tuple[int, int, int, int] | None
        ) = None,
        line_width: int = 2,
        use_full_page: bool | Literal["OnlyBoxed"] = False,
        full_page: Image.Image | None = None,
    ) -> Image.Image:
        """
        Genera un collage con las cajas seleccionadas y dibuja sus polígonos encima.
        Si represent_mbr=True, añade también el mínimo rectángulo rotado de cada polígono.
        """
        selected_ids = list(image_box_ids)
        if not selected_ids:
            raise ValueError(
                "No se puede representar una secuencia vacía de image_box_ids."
            )

        selected_boxes = [self.image_boxes[box_id] for box_id in selected_ids]
        fill_background = (
            background_color if background_color is not None else self.background_color
        )

        if use_full_page == "OnlyBoxed":
            background = compose_collage(
                list(self.image_boxes.values()), fill_background
            )
            origin_x = int(
                min(box.polygon.bounds[0] for box in self.image_boxes.values())
            )
            origin_y = int(
                min(box.polygon.bounds[1] for box in self.image_boxes.values())
            )
        elif use_full_page:
            if full_page is None:
                raise ValueError(
                    "Si se quiere usar la página completa como fondo, se debe pasar como argumento en full_page."
                )
            background = full_page.convert("RGB")

            origin_x = 0
            origin_y = 0

        else:
            # solamente la región que tiene cajas
            background = compose_collage(selected_boxes, fill_background)
            origin_x = int(min(box.polygon.bounds[0] for box in selected_boxes))
            origin_y = int(min(box.polygon.bounds[1] for box in selected_boxes))

        # Usamos el mismo anclaje que compose_collage para convertir coordenadas globales a locales.

        draw = ImageDraw.Draw(background)

        for box in selected_boxes:
            if represent_polygon:
                polygon_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in box.polygon.exterior.coords
                ]
                draw.line(polygon_points, fill=polygon_color, width=line_width)

            if represent_mbr:
                mbr_points = [
                    (float(x - origin_x), float(y - origin_y))
                    for x, y in box.polygon.minimum_rotated_rectangle.exterior.coords
                ]
                draw.line(mbr_points, fill=mbr_color, width=line_width)

        return background

    @classmethod
    def from_paragraphs(
        cls,
        paragraphs: list[Paragraph],
        task_id: int,
        background_color: tuple[int, int, int],
        last_update_time: str,
        completer: str,
        updater: str,
        annotation_unique_id: int,
        line_separator: str,
        process_images: bool,
    ) -> "AnnotatedPage":
        """
        Construye una instancia de AnnotatedPage a partir de una lista de párrafos (de dos instancias ya generadas).
        """
        instance: AnnotatedPage = cls.__new__(cls)

        instance.task_id = task_id
        instance.background_color = background_color
        instance.last_update_time = last_update_time
        instance.completer = completer
        instance.updater = updater
        instance.annotation_unique_id = annotation_unique_id
        instance.line_separator = line_separator
        instance.process_images = process_images

        instance.paragraphs = paragraphs
        instance.image_boxes = {}
        instance.text_fragments = {}

        sindex = 0
        for paragraph_index, paragraph in enumerate(instance.paragraphs):
            paragraph.index = paragraph_index
            for box in paragraph.image_boxes:
                instance.image_boxes[box.id] = box
            for fragment in paragraph.text_fragments:
                fragment.starting_index = sindex
                sindex += len(fragment.text) + 1
                instance.text_fragments[fragment.id] = fragment

        instance.__graph = instance._build_intersection_graph()

        return instance

    @staticmethod
    def combine_annotations(*annotations: "AnnotatedPage") -> "AnnotatedPage":
        """
        Combina dos anotaciones de una misma tarea ordenando sus párrafos.
        Emplean como orden de lectura de párrafos el dado por el centroide de su primera línea.
        """

        if not annotations:
            raise ValueError("Debe pasarse alguna anotación para combinar.")

        if not (len(set(ann.task_id for ann in annotations)) == 1):
            raise ValueError(
                "No se pueden combinar anotaciones de tareas diferentes: "
                f" {set(ann.task_id for ann in annotations)}"
            )

        if not (len(set(ann.line_separator for ann in annotations)) == 1):
            raise ValueError(
                "No se pueden combinar anotaciones con separadores de línea diferentes: "
                f" {set(ann.line_separator for ann in annotations)}"
            )

        combined_paragraphs = sum((ann.paragraphs for ann in annotations), start=[])

        def _topmost_order(paragraph: Paragraph) -> tuple[float, float]:
            topmost_box = paragraph.image_boxes[0]
            c = topmost_box.centroid()
            return (c[1], c[0])

        combined_paragraphs.sort(key=_topmost_order)
        last_update_time = max(ann.last_update_time for ann in annotations)

        completer = "+".join(set(ann.completer for ann in annotations))
        updater = "+".join(set(ann.updater for ann in annotations))
        annotation_id = int("000".join(str(ann.task_id) for ann in annotations))
        process_images = all(ann.process_images for ann in annotations)
        line_separator = annotations[0].line_separator

        combined_ann = AnnotatedPage.from_paragraphs(
            paragraphs=combined_paragraphs,
            task_id=annotations[0].task_id,
            background_color=annotations[0].background_color,
            last_update_time=last_update_time,
            completer=completer,
            updater=updater,
            annotation_unique_id=annotation_id,
            process_images=process_images,
            line_separator=line_separator,
        )
        combined_ann._correct_text_and_set_sindices()
        return combined_ann
