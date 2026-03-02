from preprocessing.ImageBox import ImageBox
from preprocessing.TextFragment import TextFragment
from preprocessing.Paragraph import Paragraph
from preprocessing.helpers.helper_to_classes import (
    get_connected_components,
    get_dominant_color,
    get_rotated_region,
    unrotate_image,
    reemplazar_latex_espaciado,
    trim_star_nodes,
    compose_collage,
)
from preprocessing.helpers.text_replacements import (
    replacements,
    replacements_envs,
    regex_replacements,
)
from parameters import BIG_BOX_THRESHOLD, min_nodes_for_big_box_removal
from shapely import Polygon, box as boxshape
from labelstudio.LabelStudioInterface import LabelStudioInterface
from display import display
import re
from PIL import Image
from paths import usernames_filepath
import json
import numpy as np

if usernames_filepath.exists():
    # si podemos evitar instanciar LSI, mejor
    ordered_usernames_LS = json.loads(usernames_filepath.read_text())
else:
    ordered_usernames_LS = LabelStudioInterface().usernames


class AnnotatedPage:
    """
    Clase que representa una única anotación. Recoge la información sobre las cajas-imagen y los fragmentos
    de texto (con sus relaciones), construye el grafo de adyacencia y ordena la información y la hace
    accesible de forma que la función augment_data tenga menor complejidad.
    """

    n_annotation_errors = 0
    warn_unrotate = True
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
    )

    def __init__(
        self,
        ann,
        img: Image.Image = None,
        unrotate: bool = False,
    ):

        if unrotate and AnnotatedPage.warn_unrotate:
            print(
                f"[!!!] Usar unrotate = True destruye la información sobre la posición del crop en la instancia de AnnotatedPage. "
                "Además, reduce la calidad de las imágenes por usar interpolación bicúbica, y esta misma interpolación introduce "
                "artefactos visuales en los bordes de la imagen. Úsese solamente en caso de revisión manual de las imágenes, y "
                "NO para el código de generación del dataset."
            )
            print(
                "También invalida la forma en la que se generan los párrafos, la transcripción y los starting_indices."
            )

        # corrige los resultados realizando las sustituciones
        results = self.correct_results(
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
                imgbox_id: ImageBox(
                    id=imgbox_id,
                    task_id=self.task_id,
                    **self.rotatedregion(imgbox_id, img, img_boxes_json, unrotate),
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

        self.setup_mappings(
            results
        )  # guardamos en cada dataclass los otros objetos que tiene asociados mediante una relación de labelstudio

        self.assert_pairing()  # nos aseguramos de que todas las imágenes tengan fragmento, y viceversa

        self.__graph: dict[str, set[str]] = (
            self.build_graph()
        )  # construimos el grafo de intersecciones entre cajas-imagen

        if self.order > min_nodes_for_big_box_removal:
            self.trim_star_nodes()

        # TODO: does this produce good results always? Does this cc_ordering and using the raw ccs cause any problems?

        # colocamos las componentes conexas siguiendo el orden de lectura.
        box_ccs = [
            [self.image_boxes[box_id] for box_id in component]
            for component in get_connected_components(self.__graph)
        ]

        # generamos los párrafos (componentes conexas con información extra), que añaden automáticamente información sobre
        # los centroides corregidos a cada caja-imagen. El sorted se ejecuta atuomáticamente, y se hace usando el orden
        # naif.
        self.paragraphs: list[Paragraph] = sorted(
            [Paragraph(box_cc) for box_cc in box_ccs]
        )

        sindex = 0  # índices de inicio de cada fragmento de texto (empleando
        for paragraph in self.paragraphs:
            for fragment in paragraph.text_fragments:
                fragment.starting_index = sindex
                sindex += len(fragment.text) + 1

        self.last_update_time = " ".join(
            ann["updated_at"].replace("Z", "").split("T")
        )  # última actualización de la tarea

        self.completer = ordered_usernames_LS[
            ann["completed_by"]
        ]  # persona que completó la tarea
        self.updater = ordered_usernames_LS[
            ann["updated_by"]
        ]  # útlima persona en actualizar la tarea

    @property
    def order(self):
        return len(self.graph)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, value):
        raise ValueError(
            "Por causas de starting_index y composición del documento, no es posible modificar el grafo!"
        )

    def setup_mappings(self, results: list):
        """
        A partir de las relaciones creadas en cada tarea de LS, genera respectivos diccionarios:
            1. img2text_rel, [box_id] -> fragment_id,
        que lleva el ID de una caja-imagen al id de un fragmento, y
            2. text2img_rel, [fragment_id] -> box_id,
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
                    AnnotatedPage.register_error()
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
                    AnnotatedPage.register_error()
                    # asociación fragmento -> fragmento (error de anotación)
                    print(f"(Task {self.task_id}) Asociación texto->texto.")
                    print(self.text_fragments[source_id].text)
                    print(self.text_fragments[target_id].text)
                    continue
                else:
                    AnnotatedPage.register_error()
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
        Compruba que todas las cajas están asociadas a un texto, y viceversa
        """

        for box_id, box in self.image_boxes.items():
            if len(box.associated_fragments) == 0:
                print(
                    f"\n\n\n(Task {self.task_id}) - La caja-imagen {box_id} no tiene texto asociado:"
                )
                AnnotatedPage.register_error()
                display(box.crop)

        for fragment_id, fragment in self.text_fragments.items():
            if len(fragment.associated_boxes) == 0:
                print(
                    f"\n\n\n(Task {self.task_id}) - El fragmento {fragment_id} no tiene caja-imagen asociada:"
                )
                AnnotatedPage.register_error()
                display(fragment.text)

    def __repr__(self):
        return f"<Annotation of task {self.task_id} of order {self.order}. Completed by {self.completer}, last updated by {self.updater} at {self.last_update_time}>"

    def rotatedregion(
        self, key: str, img: Image.Image, img_boxes_json, unrotate=False
    ) -> dict[str, Image.Image | Polygon | bool]:
        """
        Calcula la región recortada y el polígono asociado para cada fragmento.
        Devuelve, en orden:
            crop: PIL.Image.Image -> imagen recortada
            polygon: Polygon -> polígono de shapely
            rotation: bool -> ángulo de rotación (manual o calculado) de la región para su lectura.
            true_rectangle: bool -> si se generó usando la herramienta rectángulo.

        Si unrotate=True, la imagen se endereza y el polígono se reconstruye
        para coincidir exactamente con las nuevas dimensiones visuales (sin márgenes).
        De cualquier forma, el enderezamiento solamente es para el proceso de revisión,
        no para la generación del dataset.
        """
        H = img_boxes_json[key]["original_height"]
        W = img_boxes_json[key]["original_width"]
        val = img_boxes_json[key]["value"]

        # Obtenemos el recorte y el polígono original (en coordenadas globales)
        crop, original_poly, rotation, polygonic = get_rotated_region(
            val, W, H, img, self.background_color
        )

        if not unrotate or not rotation:
            return {
                "crop": crop,
                "polygon": original_poly,
                "rotation": rotation,
                "true_rectangle": not polygonic,
                "unrotated": False,
            }
        else:
            # Si des-rotamos, la bounding box del polígono original (rotado) suele ser
            # más grande que la imagen enderezada final, generando espacios en blanco en el collage.
            # Procedemos a crear un nuevo polígono ajustado al píxel.

            # Generamos primero la imagen final para obtener sus dimensiones reales (w, h)
            # Esto elimina las zonas transparentes sobrantes tras la rotación.
            final_crop = unrotate_image(crop, rotation)
            cw, ch = final_crop.size

            # calculamos el punto de anclado basándonos en la geometría original.
            # Usamos el mínimo rectángulo rotado para hallar la verdadera esquina
            # superior izquierda visual, independientemente de la orientación de los ejes.
            rotated_rect = original_poly.minimum_rotated_rectangle
            rect_coords = list(rotated_rect.exterior.coords)[:-1]

            # ordenamos vértices: prioridad menor Y (arriba), desempate menor X (izquierda) - extremadamente improbable salvo en el borde inferior
            pivot_point = sorted(rect_coords, key=lambda p: (p[1], p[0]))[0]
            pivot_x, pivot_y = pivot_point

            # construimos un nuevo polígono rectangular
            # empieza en el pivote original pero tiene exactamente las dimensiones de la imagen recortada.
            # esto nos asegura consistencia al pegar en el lienzo del collage.
            new_poly = boxshape(pivot_x, pivot_y, pivot_x + cw, pivot_y + ch)

            return {
                "crop": final_crop,
                "polygon": new_poly,
                "rotation": rotation,
                "true_rectangle": not polygonic,
                "unrotated": True,
            }

    def build_graph(self):
        """
        Genera el grafo de intersecciones de una anotación.
        Devuelve un diccionario de adyacencia {id: set(id_adyacentes)}.
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
    def correct_results(results):
        """
        Realiza las sustituciones especificadas en 'replacements' y 'replacements_envs' en
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

    def adjacency_matrix(self) -> np.ndarray:

        adjacency_mx = np.zeros((self.order, self.order))

        box_ids = list(self.graph.keys())

        for i, j in np.ndindex(adjacency_mx.shape):
            i_id, j_id = box_ids[i], box_ids[j]
            if i_id in self.graph[j_id]:
                adjacency_mx[i, j] = 1
                adjacency_mx[j, i] = 1  # por simetría

        return adjacency_mx

    def generate_collage(self, box_id_sequence: set[str] | list[str]) -> Image.Image:
        """
        Genera el collage de recortes para una secuencia de ids de cajas (un subgrafo).
        """
        if not isinstance(box_id_sequence, set):
            if len(box_id_sequence) != len(set(box_id_sequence)):
                raise ValueError("Hay cajas-imagen repetidas en generate_collage()")
            box_id_sequence = set(box_id_sequence)

        subgraph_image_boxes = [self.image_boxes[box_id] for box_id in box_id_sequence]

        return compose_collage(subgraph_image_boxes, self.background_color)

    def trim_star_nodes(
        self,
        relative_threshold: float = BIG_BOX_THRESHOLD,
    ) -> None:

        self.__graph = trim_star_nodes(self.graph, relative_threshold)

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
    def is_single_paragraph(self):
        return len([paragraph for paragraph in self.paragraphs if len(paragraph)]) == 1

    @staticmethod
    def register_error():
        AnnotatedPage.n_annotation_errors += 1
