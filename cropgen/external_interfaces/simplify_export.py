import json
from pathlib import Path

from cropgen.shared.PathBundle import PathBundle
import os
from cropgen.shared.LSTypedDicts.results import (
    TextCorrectionResult,
    TextRegionResult,
    RelationResult,
    RectangleResult,
    PolygonResult,
)
from cropgen.shared.LSTypedDicts.aggregates import (
    LabelStudioTask,
    RawAnnotation,
    ResultItem,
)
from cropgen.shared.LSTypedDicts.simplified import (
    SimplifiedTask,
    SimplifiedAnnotation,
    SimplifiedTextCorrectionResult,
    SimplifiedResultItem,
)
from typing import List


def resolve_text_for_group(group: List[ResultItem], full_text: str) -> list[str]:
    """
    Escoge el texto final (corrección si hay corrección, texto original si no)
    de una relación. Emplea que los resultados de un mismo "grupo" tienen el
    mismo ID, usa, en orden de prioridad:
    Correcciones manuales > Texto en la etiqueta > Texto entre start/end.
    """
    # mira si hay corrección (busca todas las ocurrencias)
    correction_res_list: list[TextCorrectionResult] = [
        r for r in group if isinstance(r, TextCorrectionResult)
    ]
    collected_corrections = []

    for res in correction_res_list:
        val = res.value.text

        if isinstance(val, list):
            # usamos extend para el caso en el que hay múltiples correcciones (errores de los anotadores)
            valid_texts = [v for v in val if isinstance(v, str) and v.strip()]
            collected_corrections.extend(valid_texts)

        elif isinstance(val, str) and val.strip():
            collected_corrections.append(val)

    if collected_corrections:
        return collected_corrections

    # busca la etiqueta original
    label_res: TextRegionResult | None = next(
        (
            TextRegionResult.model_validate(r)
            for r in group
            if r.type in ["labels", "hypertextlabels"]
        ),
        None,
    )
    if label_res:
        val = label_res.value
        # busca el texto explícito
        if val.text:
            t = val.text
            # Devuelve lista directamente si es lista, o encapsula si es string
            return t if isinstance(t, list) else [t]

        # busca rangos start-end
        if full_text:
            try:
                return [full_text[int(val.start) : int(val.end)]]
            except:
                print(
                    "Error durante la formación de lo grupos: rango start-end inválido (no son ints)"
                )
                pass
    return []


def convert_result_raw(
    obj: (
        dict
        | RelationResult
        | TextCorrectionResult
        | PolygonResult
        | RectangleResult
        | TextRegionResult
    ),
) -> (
    RelationResult
    | PolygonResult
    | TextCorrectionResult
    | RectangleResult
    | TextRegionResult
):
    if not isinstance(obj, dict):
        return obj
    match obj["type"]:
        case "relation":
            return RelationResult.model_validate(obj)
        case "polygonlabels":
            return PolygonResult.model_validate(obj)
        case "textarea":
            return TextCorrectionResult.model_validate(obj)
        case "rectanglelabels":
            return RectangleResult.model_validate(obj)
        case "labels" | "hypertextlabels":
            return TextRegionResult.model_validate(obj)
        case _:
            raise ValueError('El diccionario no tiene ["type"] conocido')


def convert_result_simplified(
    obj: (
        dict
        | RelationResult
        | PolygonResult
        | SimplifiedTextCorrectionResult
        | RectangleResult
        | TextRegionResult
    ),
) -> (
    RelationResult
    | PolygonResult
    | SimplifiedTextCorrectionResult
    | RectangleResult
    | TextRegionResult
):
    if not isinstance(obj, dict):
        return obj

    match obj["type"]:
        case "relation":
            return RelationResult.model_validate(obj)
        case "polygonlabels":
            return PolygonResult.model_validate(obj)
        case "textarea":
            return SimplifiedTextCorrectionResult.model_validate(obj)
        case "rectanglelabels":
            return RectangleResult.model_validate(obj)
        case "labels" | "hypertextlabels":
            return TextRegionResult.model_validate(obj)
        case _:
            raise ValueError('El diccionario no tiene ["type"] conocido')


def simplify_export(raw_export_filepath: Path, simplified_filepath: Path) -> None:
    """
    Función de simplificación del export.json. Toma el archivo, y produce el archivo
    de salida.
    """

    if not raw_export_filepath.exists():
        raise FileNotFoundError(
            f"Error: No existe el archivo en la ruta {raw_export_filepath=}. Llama download_updated_export({raw_export_filepath})"
        )

    if isinstance(simplified_filepath, str):
        simplified_filepath = Path(simplified_filepath)

    os.makedirs(os.path.dirname(simplified_filepath), exist_ok=True)

    try:
        tasks: list[LabelStudioTask] = [
            LabelStudioTask.model_validate(task_dict)
            for task_dict in json.loads(raw_export_filepath.read_text(encoding="utf-8"))
        ]

    except Exception as e:
        raise ValueError(f"Error reading JSON: {e}")

    processed_tasks: List[SimplifiedTask] = []

    for task in tasks:

        # definimos la nueva estructura de las tareas
        new_task = task.model_copy()
        full_text = task.data.transcription
        annotations: list[RawAnnotation] = task.annotations
        empty_boxes_on_page = 0
        simplified_annotations: list[SimplifiedAnnotation] = []
        for ann in annotations:
            new_results: list[SimplifiedResultItem] = []
            raw_results = ann.result
            # agrupamos resultados por su ID
            results_by_id: dict[str, list[ResultItem]] = {}
            relations = []
            for res in (convert_result_raw(r) for r in raw_results):

                if isinstance(res, RelationResult):
                    relations.append(res)
                    continue

                rid = res.id
                if rid not in results_by_id:
                    results_by_id[rid] = []

                results_by_id[rid].append(res)

            # iteramos sobre cada ID
            for rid, group in results_by_id.items():
                # buscamos solamente cajas-imagen
                box_res = None
                for item in group:
                    if item.type == "rectanglelabels":
                        box_res = RectangleResult.model_validate(item)
                        break
                    elif item.type == "polygonlabels":
                        box_res = PolygonResult.model_validate(item)
                        break

                if box_res is not None:
                    box_res: PolygonResult | RectangleResult
                    new_results.append(box_res)
                else:
                    final_text_list = resolve_text_for_group(group, full_text)
                    if not final_text_list:
                        empty_boxes_on_page += 1
                        continue
                    if len(final_text_list) > 1:
                        label_res = next(
                            (
                                TextRegionResult.model_validate(r)
                                for r in group
                                if r.type in ["labels", "hypertextlabels"]
                            ),
                            None,
                        )
                        original_text = (
                            str(label_res.value.text) if label_res is not None else "<sin etiqueta original>"
                        )
                        print(
                            f"En la tarea {task.id}, el fragmento \n\t > {original_text.replace('\n', '\n\t')}\n tiene múltiples correcciones:"
                        )
                        for correction in final_text_list:
                            print(f"\t > {correction.replace('\n', '\n\t')}")
                    # en el caso en el que haya más de una, cogemos la primera (placeholder)
                    final_text = final_text_list[0]
                    if final_text and final_text.strip():
                        synthetic_res = SimplifiedTextCorrectionResult.model_validate(
                            {
                                "id": rid,
                                "type": "textarea",
                                "value": {"text": [final_text]},
                                "from_name": "text_adapter",
                                "to_name": "image",
                            }
                        )
                        new_results.append(synthetic_res)
                    else:
                        empty_boxes_on_page += 1
            # añadimos las relaciones
            new_results.extend(relations)
            new_ann = SimplifiedAnnotation.model_validate(
                {**ann.model_dump(), "result": new_results}
            )
            simplified_annotations.append(new_ann)
        # creamos la nueva tarea simplificada con las anotaciones ya procesadas
        new_simplified_task = SimplifiedTask.model_validate(
            {**new_task.model_dump(), "annotations": simplified_annotations}
        )
        processed_tasks.append(new_simplified_task)
        if empty_boxes_on_page > 0:
            print(
                f"Tarea {task.id}: Se han ignorado {empty_boxes_on_page} fragmentos de texto vacíos (o solo con espacios)."
            )

    simplified_filepath.write_text(
        json.dumps(
            [t.model_dump() for t in processed_tasks], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"Simplificadas {len(processed_tasks)} tareas.")
    print(f"Se han guardado en {simplified_filepath}")


def load_simplified_export(paths: PathBundle) -> List[SimplifiedTask]:
    """
    Carga el archivo export.json y devuelve la lista de tareas.
    """
    path = paths.simplified_filepath

    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Error leyendo el archivo {path=}. {e}")

    if isinstance(data, list):
        return data  # type: ignore

    if isinstance(data, dict):  # buscamos los diferentes resultados
        for k in ("tasks", "data", "results"):
            if isinstance(data.get(k), list):
                return data[k]  # type: ignore

    raise ValueError(
        "El tipo de archivo que se ha especificado no tiene la estructura esperada."
    )
