from cropgen.processing.ImageBox import ImageBox
from cropgen.processing.TextFragment import TextFragment
from collections.abc import Iterable
from cropgen.processing.helpers.PairingErrors import (
    NoAssociationError,
    MultipleAssociationError,
    RepeatedSameAssociationError,
    SameToSameAssociation,
)
from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.helpers.helper_to_classes import (
    get_connected_components,
    get_dominant_color,
    reemplazar_latex_espaciado,
    compose_collage,
    subdictionary,
)
from cropgen.processing.helpers.text_replacements import (
    replacements,
    replacements_envs,
    regex_replacements,
)
from cropgen.shared.default_parameters import (
    big_box_threshold,
    min_nodes_for_big_box_removal,
)
from cropgen.shared.display import display
import re
from PIL import Image
import numpy as np


class AnnotatedPage:
    """
    Clase que representa una única anotación. Recoge la información sobre las cajas-imagen y los fragmentos
    de texto (con sus relaciones), construye el grafo de adyacencia, crea los párrafos y ordena la información
    y la hace accesible de forma que la función augment_data tenga menor complejidad.
    """

    n_annotation_errors = 0
    warn_unrotate = True
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
    )

    def __init__(
        self,
        ann: dict[str, dict | str | int | list[str]],
        img: Image.Image = None,
        unrotate: bool = False,
        usernames_labelstudio: list[str] = None,
    ):

        if unrotate and AnnotatedPage.warn_unrotate:
            print(
                "[!!!] Usar unrotate = True destruye la información sobre la posición del crop en la instancia de AnnotatedPage. "
                "Además, reduce la calidad de las imágenes por usar interpolación bicúbica, y esta misma interpolación introduce "
                "artefactos visuales en los bordes de la imagen. Úsese solamente en caso de revisión manual de las imágenes, y "
                "NO para el código de generación del dataset."
            )
            print(
                "También invalida la forma en la que se generan los párrafos, la transcripción y los starting_indices."
            )
        assert (
            usernames_labelstudio is not None
        ), "Es necesario proporcionar la lista de usernames de LS para generar la anotación."

        # corrige los resultados realizando las sustituciones
        results: list = self._correct_results(
            ann.get("result", [])
        )  # resultados de la anotación (diccionario muy grande con un poco de toodo)
        self.task_id = int(ann["task"])

        self.background_color = get_dominant_color(img)

        res_map = {r.get("id"): r for r in results}  # id -> dato anotado

        img_boxes_json = {
            rid: r
            for rid, r in res_map.items()
            if r.get("type") in ("rectanglelabels", "polygonlabels")
        }

        self.image_boxes: dict[str, ImageBox] = (
            {  # conjunto de cajas-imagen (instancias de ImageBox)
                imgbox_id: ImageBox.from_json_value(
                    img_boxes_json[imgbox_id], imgbox_id, self.task_id, img, unrotate
                )
                for imgbox_id in img_boxes_json
            }
        )

        txt_boxes_json: dict[str, dict] = {
            rid: r
            for rid, r in res_map.items()
            if r.get("type") in ("labels", "hypertextlabels", "textarea")
        }

        self.text_fragments: dict[str, TextFragment] = (
            {  # conjunto de fragmentos de texto (instancias de TextFragment)
                fragment_id: TextFragment(
                    id=fragment_id,
                    text=(
                        txtbox_res["value"]["text"].strip()
                        if isinstance(txtbox_res["value"]["text"], str)
                        else " ".join(txtbox_res["value"]["text"]).strip()
                    ),
                    task_id=self.task_id,
                )
                for (fragment_id, txtbox_res) in txt_boxes_json.items()
            }
        )

        self._setup_mappings(
            results
        )  # guardamos en cada dataclass los otros objetos que tiene asociados mediante una relación de external_interfaces

        self.assert_pairing()  # nos aseguramos de que todas las imágenes tengan fragmento, y viceversa

        self.__graph: dict[str, set[str]] = (
            self._build_intersection_graph()
        )  # construimos el grafo de intersecciones entre cajas-imagen

        if self.order > AnnotatedPage.min_nodes_for_big_box_removal:
            self.trim_star_nodes()

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

        sindex = 0  # índices de inicio de cada fragmento de texto

        # notemos que solamente las imágenes que estén en un párrafo tienen sindex...
        for paragraph_index, paragraph in enumerate(self.paragraphs):
            paragraph.index = paragraph_index
            for fragment in paragraph.text_fragments:
                fragment.starting_index = sindex
                sindex += len(fragment.text) + 1

        self.last_update_time = " ".join(
            ann["updated_at"].replace("Z", "").split("T")
        )  # última actualización de la tarea

        completer_index = ann["completed_by"]
        updater_index = ann["updated_by"]
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
        self.annotation_unique_id = ann["id"]

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

    def _setup_mappings(self, results: list) -> None:
        """
        A partir de las relaciones creadas en cada tarea de LS, genera respectivos diccionarios:
            1. img2text_rel: dict[str, str], box_id -> fragment_id,
        que lleva el ID de una caja-imagen al id de un fragmento, y
            2. text2img_rel: dict[str, str], fragment_id -> box_id,
        que lleva el ID de un fragmento a su caja-imagen correspondiente.
        """

        for r in results:
            if r.get("type") == "relation":  # si el resultado es una relación
                source_id, target_id = r.get("from_id"), r.get("to_id")

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

    @staticmethod
    def _correct_results(results: list) -> list:
        """
        Realiza las sustituciones especificadas en 'replacements', 'replacements_envs' y 'replacements_regex' en
        los resultados de una tarea (ambas son variables de clase).
        """
        # return results

        for r in results:
            # solamente hacemos las sustituciones en los fragmentos de texto:
            if r.get("type") in ("labels", "hypertextlabels", "textarea"):

                # teóricamente esto debería ser únicamente un elemento, pero no realizamos suposiciones
                # innecesarias
                text_res = r["value"]["text"]
                if isinstance(text_res, list):

                    for i in range(len(text_res)):
                        text = text_res[i]
                        for old, new in replacements:
                            newtext = text.replace(
                                old, new
                            )  # hacemos todos los cambios indicados
                            text = newtext

                        for beg, end in replacements_envs:
                            text = reemplazar_latex_espaciado(text, beg, end)

                        for pattern, substitution in regex_replacements:
                            text = re.sub(pattern, substitution, text)

                        text_res[i] = text

                    r["value"][
                        "text"
                    ] = text_res  # lo sustituímos por el original (en la variable, no en el .json)

                elif isinstance(text_res, str):

                    for old, new in replacements:
                        text_res = text_res.replace(
                            old, new
                        )  # hacemos todos los cambios indicados

                    for beg, end in replacements_envs:
                        text_res = reemplazar_latex_espaciado(text_res, beg, end)

                    for pattern, substitution in regex_replacements:
                        text_res = re.sub(pattern, substitution, text_res)

                    # para terminar, quitamos un <.strip> para eliminar los espacios que Malgoire haya podido
                    # añadir al princpio o final.
                    r["value"]["text"] = text_res.strip()

        return results

    def generate_collage(
        self,
        box_id_sequence: set[str] | list[str],
        background_color: tuple | None = None,
    ) -> Image.Image:
        """
        Genera el collage de recortes para una secuencia de ids de cajas (un subgrafo), colocando en sus posiciones en
        la página original los recortes, rellenando el resto con el color promedio de la imagen y recortando la imagen
        al tamaño mínimo que contiene todos los recortes colocados.
        """
        if not isinstance(box_id_sequence, set):
            if len(box_id_sequence) != len(set(box_id_sequence)):
                raise ValueError("Hay cajas-imagen repetidas en generate_collage()")
            box_id_sequence = set(box_id_sequence)

        subgraph_image_boxes = [self.image_boxes[box_id] for box_id in box_id_sequence]

        return compose_collage(
            subgraph_image_boxes,
            background_color if background_color else self.background_color,
        )

    def trim_star_nodes(
        self,
        relative_threshold: float = big_box_threshold,
    ) -> None:
        """
        Elimina los nodos con una conectividad mayor a relative_threshold
        """

        adj_graph = self.__graph.copy()
        nodes_to_remove = []

        for rid, neighbors in adj_graph.items():
            if (
                len(neighbors) / (len(adj_graph) - 1) > relative_threshold
            ):  # si pasa el umbral
                nodes_to_remove.append(rid)

        for rid in nodes_to_remove:
            self.image_boxes[rid].fragment.starting_index = -1

            del adj_graph[rid]  # quitamos el nodo en forma de estrella

            for other in adj_graph:  # eliminamos todas sus referencias
                adj_graph[other].discard(rid)

        self.__graph = adj_graph

    def cluster_reading_order(
        self, box_ids: list["str"]
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
        fragments = sorted(fragments, key=lambda x: x.starting_index)

        transcription = " ".join([fragment.text for fragment in fragments])
        sindex = fragments[0].starting_index

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
            2. Son fragmentos que tenían una conectividad muy alta y se han desconectado usando trim_star_nodes
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

    def get_average_rotation(self, img_box_ids: Iterable[str]):
        """
        Devuelve la rotación promedio de un grupo de cajas-imagen dados sus ids.
        """
        image_boxes = [self.image_boxes[box_id] for box_id in img_box_ids]
        areas = [box.polygon.area for box in image_boxes]
        angles_in_radians = [np.radians(box.rotation) for box in image_boxes]

        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))

        return -np.degrees(np.arctan2(sum_sin, sum_cos))
