from preprocessing.classes.ImageBox import ImageBox
from preprocessing.classes.TextFragment import TextFragment
from preprocessing.classes.helper_to_classes import (
    get_connected_components,
    get_dominant_color,
    get_rotated_region,
    get_union_rect,
    unrotate_image,
    reemplazar_latex_espaciado,
    trim_star_nodes,
)
from parameters import BIG_BOX_THRESHOLD, min_nodes_for_big_box_removal
from shapely import box as boxshape
from labelstudio.LabelStudioInterface import LabelStudioInterface
from display import display
import re
from PIL import Image
from paths import simplified_filepath, usernames_filepath
import json

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

    warn_unrotate = True
    replacements = [
        (r"\'e", "é"),
        (r"\^e", "ê"),
        (r"\`e", "è"),
        (r"\"e", "ë"),
        (r"\'a", "á"),
        (r"\^a", "ä"),
        (r"\`a", "à"),
        (r"\"a", "ä"),
        (r"\'i", "í"),
        (r"\^i", "î"),
        (r"\`i", "ì"),
        (r"\"i", "ï"),
        (r"\'o", "ò"),
        (r"\^o", "ô"),
        (r"\`o", "ò"),
        (r"\"o", "ö"),
        (r"\'u", "ú"),
        (r"\^u", "û"),
        (r"\`u", "ù"),
        (r"\"u", "ü"),
        (r"\c{c}", "ç"),
        (r"\smallskip", ""),
        (r"\medskip", ""),
        (r"\bigskip", ""),
        (r"\begin{equation}", "$"),
        (r"\begin{equation}", "$"),
        (r"\end{equation}", "$"),
        (r"\begin{equation*}", "$"),
        (r"\end{equation*}", "$"),
        (r"\(", "$"),
        (r"\)", "$"),
        (r"\[", "$"),
        (r"\]", "$"),
        (r"$$", "$"),
        (r"\dots", "..."),
        (r"\quad", ""),
        (r"\colon", ":"),
        (r"\,", " "),
        (r"\;", " "),
        (r"\ ", " "),
        (r"\etale", "étale"),
        (" e0", "à"),
        (" e9", "é"),
        (" f9", "ù"),
        (r"\{\mathcal U\}", r"\mathcal U"),
        (r"\{\mathcal{U}\}", r"\mathcal U"),
    ]
    replacements_envs = [
        # la sintaxis es (apertura, cierre) del entorno, se quita t0do menos lo de dentro
        (r"{\bf", "}"),
        (r"{ \bf", "}"),
        (r"{\it", "}"),
        (r"{ \it", "}"),
        (r"{\sl", "}"),
        (r"{ \sl", "}"),
        (r"\textit{", "}"),
        (r"\textbf{", "}"),
        (r"\textsl{", "}"),
        (r"\underline{", "}"),
        (r"\emph{", "}"),
        (r"\footnote{", "}"),
        (r"\begin{center}", r"\end{center}"),
    ]

    regex_replacements = [
        (
            r"\\n(?!(?:ot|ew|ode|u|eq|exists|ewpage|oindent|natural|eg|earrow|warrow|abla)\b)",
            " ",
        ),
        # reemplazar nuevas líneas
        (r"\\U\b", r"\\mathcal U"),
        (r"\\E\b", r"\\mathcal E"),
    ]
    __slots__ = (
        "background_color",
        "image_boxes",
        "text_fragments",
        "task_id",
        "__graph",
        "last_update_time",
        "completer",
        "updater",
        "__cc_ordering",
        "ordered_connected_components",
    )

    def __init__(self, ann, task_id, img, cc_ordering: bool, unrotate: bool = False):

        if unrotate and AnnotatedPage.warn_unrotate:
            print(
                f"Usar unrotate = True destruye la información sobre la posición del crop en la instancia de AnnotatedPage. "
                "Además, reduce la calidad de las imágenes por usar interpolación bicúbica, y esta misma interpolación introduce "
                "artefactos visuales en los bordes de la imagen. Úsese solamente en caso de revisión manual de las imágenes, y "
                "NO para el código de generación del dataset."
            )
        # corrige los resultados realizando las sustituciones
        results = self.correct_results(
            ann.get("result", [])
        )  # resultados de la anotación (diccionario muy grande con un poco de toodo)
        self.task_id = int(task_id)

        self.background_color = get_dominant_color(img)

        res_map = {r.get("id"): r for r in results}  # id -> dato anotado

        img_boxes_json = {
            rid: r
            for rid, r in res_map.items()
            if r.get("type") in ("rectanglelabels", "polygonlabels")
        }

        self.image_boxes = {  # conjunto de cajas-imagen (instancias de ImageBox)
            imgbox_id: ImageBox(
                id=imgbox_id,
                task_id=self.task_id,
                **self.rotatedregion(imgbox_id, img, img_boxes_json, unrotate),
            )
            for imgbox_id in img_boxes_json
        }

        txt_boxes_json = {
            rid: r
            for rid, r in res_map.items()
            if r.get("type") in ("labels", "hypertextlabels", "textarea")
        }

        self.text_fragments = (
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

        self.__graph = (
            self.build_graph()
        )  # construimos el grafo de intersecciones entre cajas-imagen

        # colocamos las componentes conexas siguiendo el orden de lectura.
        cc = get_connected_components(self.__graph)

        for index, component in enumerate(cc):  # orden intra-componente
            cc[index] = self.reading_order(
                component, False
            )  # lo ordenamos en orden de lectura

        cc = sorted(
            cc,
            key=lambda comp: (  # los ordenamos usando el orden naif: el "párrafo" que empiece antes va antes.
                self.image_boxes[comp[0]].polygon.bounds[1],
                self.image_boxes[comp[0]].polygon.bounds[0],
            ),
        )

        self.ordered_connected_components = cc

        self.__cc_ordering = cc_ordering
        self.assign_starting_indices()  # asignamos los sindex a cada fragmento

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
    def cc_ordering(self) -> bool:
        return self.__cc_ordering

    @cc_ordering.setter
    def cc_ordering(self, value: bool) -> None:
        if value == self.cc_ordering:
            pass
        else:
            self.__cc_ordering = value
            print(
                "El orden interno de los fragmentos para la anotación ha cambiado. Redefiniendo los starting_index de cada instancia de TextFragment."
            )
            self.assign_starting_indices()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, value):

        self.__graph = value
        cc = get_connected_components(self.__graph)

        for index, component in enumerate(cc):  # orden intra-componente
            cc[index] = self.reading_order(
                component, False
            )  # lo ordenamos en orden de lectura

        cc = sorted(
            cc,
            key=lambda comp: (  # los ordenamos usando el orden naif: el "párrafo" que empiece antes va antes.
                self.image_boxes[comp[0]].polygon.bounds[1],
                self.image_boxes[comp[0]].polygon.bounds[0],
            ),
        )
        self.ordered_connected_components = cc

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
                    # asociación fragmento -> fragmento (error de anotación)
                    print(f"(Task {self.task_id}) Asociación texto->texto.")
                    print(self.text_fragments[source_id].text)
                    print(self.text_fragments[target_id].text)
                    continue
                else:
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
                display(box.crop)

        for fragment_id, fragment in self.text_fragments.items():
            if len(fragment.associated_boxes) == 0:
                print(
                    f"\n\n\n(Task {self.task_id}) - El fragmento {fragment_id} no tiene caja-imagen asociada:"
                )
                display(fragment.text)

    def __repr__(self):
        return f"<Annotation of task {self.task_id} of order {self.order}. Completed by {self.completer}, last updated by {self.updater} at {self.last_update_time}>"

    def rotatedregion(self, key, img, img_boxes_json, unrotate=False):
        """
        Calcula la región recortada y el polígono asociado para cada fragmento.
        Devuelve, en orden:
            crop: Image -> imagen recortada
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
                        for old, new in AnnotatedPage.replacements:
                            newtext = text.replace(
                                old, new
                            )  # hacemos todos los cambios indicados
                            text = newtext

                        for beg, end in AnnotatedPage.replacements_envs:
                            text = reemplazar_latex_espaciado(text, beg, end)

                        for pattern, substitution in AnnotatedPage.regex_replacements:
                            text = re.sub(pattern, substitution, text)

                        text_res[i] = text

                    r["value"][
                        "text"
                    ] = text_res  # lo sustituímos por el original (en la variable, no en el .json)

                elif isinstance(text_res, str):

                    for old, new in AnnotatedPage.replacements:
                        text_res = text_res.replace(
                            old, new
                        )  # hacemos todos los cambios indicados

                    for beg, end in AnnotatedPage.replacements_envs:
                        text_res = reemplazar_latex_espaciado(text_res, beg, end)

                    for pattern, substitution in AnnotatedPage.regex_replacements:
                        text_res = re.sub(pattern, substitution, text_res)

                    # para terminar, quitamos un <.strip> para eliminar los espacios que Malgoire haya podido
                    # añadir al princpio o final.
                    r["value"]["text"] = text_res.strip()

        return results

    def adjacency_matrix(self):
        import numpy as np

        adjacency_mx = np.zeros((self.order, self.order))

        box_ids = list(self.graph.keys())

        for i, j in np.ndindex(adjacency_mx.shape):
            i_id, j_id = box_ids[i], box_ids[j]
            if i_id in self.graph[j_id]:
                adjacency_mx[i, j] = 1
                adjacency_mx[j, i] = 1  # por simetría

        return adjacency_mx

    def generate_collage(self, box_id_sequence: set[str] | list[str]):
        """
        Genera el collage de recortes para una secuencia de ids de cajas (un subgrafo).
        """
        # 1. Obtenemos los polígonos de la secuencia solicitada

        if not isinstance(box_id_sequence, set):
            box_id_sequence = set(box_id_sequence)

        subgraph_image_boxes = [self.image_boxes[box_id] for box_id in box_id_sequence]

        # 2. Calculamos la mínima región de la imagen que contiene todas las cajas
        X1, Y1, X2, Y2 = get_union_rect([box.polygon for box in subgraph_image_boxes])

        # Convertimos a enteros (Floor para arriba-izq, Ceil para abajo-der para asegurar cobertura)
        X1, Y1 = int(X1), int(Y1)
        X2, Y2 = int(X2) + 1, int(Y2) + 1

        crop_width, crop_height = X2 - X1, Y2 - Y1

        # 3. Creamos el lienzo
        collage = Image.new(
            "RGB", (crop_width, crop_height), tuple(self.background_color)
        )

        # 4. Pegamos las imágenes (respetando el orden dado en box_id_sequence)

        for box in subgraph_image_boxes:
            box_x0, box_y0, _, _ = box.polygon.bounds

            # Calculamos posición relativa al nuevo lienzo
            paste_x, paste_y = int(box_x0 - X1), int(box_y0 - Y1)

            if box.crop.mode == "RGBA":
                # Usamos la propia imagen como máscara de transparencia
                collage.paste(box.crop, (paste_x, paste_y), mask=box.crop)
            else:
                collage.paste(box.crop, (paste_x, paste_y))

        return collage

    def reading_order(self, box_id_sequence, cc_ordering: bool = None):
        """Dado un grupo de ids de cajas, los pone en orden de lectura.
        Si self.cc_ordering, los fragmentos de una misma componente conexa siempre se colocan seguidos (en el orden de lectura).

        Si no, se usa el orden de lectura 'naif', arriba-abajo izquierda-derecha (el mismo que se impone dentro de cada componente conexa)
        """
        if cc_ordering is None:
            cc_ordering = self.cc_ordering

        if cc_ordering:
            assert set(box_id_sequence) == set(
                self.image_boxes.keys()
            ), "Para usar cc_ordering = True, deben pasarse todas las ids de las cajas."

        if not cc_ordering:
            return sorted(
                list(box_id_sequence),
                key=lambda box_id: (
                    self.image_boxes[box_id].polygon.bounds[1],
                    self.image_boxes[box_id].polygon.bounds[0],
                ),
            )

        return [
            box_id
            for component in self.ordered_connected_components
            for box_id in component
        ]  # aplanamos la lista de listas ordenadamente

    def concatenate_transcritions(
        self, box_id_sequence, cc_ordering=None, return_first_sindex: bool = False
    ):
        """
        Concatena las transcripciones correspondientes a un grupo de cajas (un subgrafo), siguiendo el orden en el que
        aparecen en la transcripción completa.
        self.cc_ordering define cómo se compone la transcripción total (funciona como en reading_order)
        """
        if not isinstance(box_id_sequence, set):
            box_id_sequence = set(box_id_sequence)

        box_id_sequence_reading_order = self.reading_order(box_id_sequence, cc_ordering)
        concatenated_transcription = " ".join(
            [
                self.image_boxes[box_id].associated_fragments[0].text
                for box_id in box_id_sequence_reading_order
            ]
        )
        if not return_first_sindex:
            return concatenated_transcription
        else:
            return (
                concatenated_transcription,
                self.image_boxes[box_id_sequence_reading_order[0]]
                .associated_fragments[0]
                .starting_index,
            )

    def assign_starting_indices(self):
        """
        Asigna los índices de comienzo (relativos a la transcripción completa) a cada instancia de TextFragment en self.text_fragments.
        Se toma self.cc_ordering en las llamadas a self.reading_order: define cómo se compone la transcripción total.
        """
        box_id_sequence_reading_order = self.reading_order(
            list(self.image_boxes.keys())
        )

        fragments_att_ordered = [
            self.image_boxes[box_id].associated_fragments
            for box_id in box_id_sequence_reading_order
        ]

        associated_fragments = [
            fragments[0] if (len(fragments) == 1) else None
            for fragments in fragments_att_ordered
        ]

        if None in associated_fragments:
            print(
                f"Tarea {self.task_id} - Hay Cajas con un número anómalo de fragmentos asociados (!= 1)."
            )

        associated_fragments = [x for x in associated_fragments if x is not None]

        sindex = 0
        for fragment in associated_fragments:
            if isinstance(fragment, TextFragment):
                fragment.starting_index = sindex
                sindex += len(fragment.text) + 1
            else:
                sindex += 1

    def trim_star_nodes(
        self,
        relative_threshold: float = BIG_BOX_THRESHOLD,
        min_nodes: int = min_nodes_for_big_box_removal,
    ) -> None:
        if self.order > min_nodes:
            self.graph = trim_star_nodes(self.graph, relative_threshold)

    def cluster_reading_order(self, box_ids: list["str"]) -> tuple[Image, str, int]:
        """
        Given a list of box_ids, returns the corresponding collage, its concatenated text, and the starting index of it all.
        """

        collage = self.generate_collage(box_ids)
        transcription, sindex = self.concatenate_transcritions(
            box_ids,
            cc_ordering=(box_ids == self.graph.keys()),
            return_first_sindex=True,
        )
        return collage, transcription, sindex
