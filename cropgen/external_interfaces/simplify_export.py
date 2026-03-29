import json
from pathlib import Path
from cropgen.shared.PathBundle import PathBundle
import os


def resolve_text_for_group(group, full_text):
    """
    Escoge el texto final (corrección si hay corrección, texto original si no)
    de una relación. Emplea que los resultados de un mismo "grupo" tienen el
    mismo ID, usa, en orden de prioridad:
    Correcciones manuales > Texto en la etiqueta > Texto entre start/end.
    """
    # mira si hay corrección (busca todas las ocurrencias)
    correction_res_list = [r for r in group if r.get("type") == "textarea"]
    collected_corrections = []

    for res in correction_res_list:
        val = res.get("value", {}).get("text", [])

        if isinstance(val, list):
            # usamos extend para el caso en el que hay múltiples correcciones (errores de los anotadores)
            valid_texts = [v for v in val if isinstance(v, str) and v.strip()]
            collected_corrections.extend(valid_texts)

        elif isinstance(val, str) and val.strip():
            collected_corrections.append(val)

    if collected_corrections:
        return collected_corrections

    # busca la etiqueta original
    label_res = next(
        (r for r in group if r.get("type") in ["labels", "hypertextlabels"]), None
    )
    if label_res:
        val = label_res.get("value", {})
        # busca el texto explícito
        if val.get("text"):
            t = val.get("text")
            # Devuelve lista directamente si es lista, o encapsula si es string
            return t if isinstance(t, list) else [t]

        # busca rangos start-end
        if "start" in val and "end" in val and full_text:
            try:
                return [full_text[int(val["start"]) : int(val["end"])]]
            except:
                print(
                    "Error durante la formación de lo grupos: rango start-end inválido (no son ints)"
                )
                pass
    return []


def simplify_export(raw_export_filepath: Path, simplified_filepath: Path):
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
        raw_data = json.loads(raw_export_filepath.read_text(encoding="utf-8"))

    except Exception as e:
        raise ValueError(f"Error reading JSON: {e}")

    # normalizamos los datos
    tasks = raw_data if isinstance(raw_data, list) else [raw_data]
    processed_tasks = []

    for task in tasks:
        # definimos la nueva estructura de las tareas
        new_task = task.copy()

        full_text = task.get("data", {}).get("transcription", "")
        annotations = task.get("annotations", [])

        empty_boxes_on_page = 0

        for ann in annotations:
            # ADD THIS LINE HERE:
            new_results = []  # <-- Reset the list for every specific annotation

            raw_results = ann.get("result", [])

            # agrupamos resultados por su ID
            results_by_id = {}
            relations = []

            for r in raw_results:
                if r.get("type") == "relation":
                    relations.append(r)
                    continue

                rid = r.get("id")
                if rid not in results_by_id:
                    results_by_id[rid] = []
                results_by_id[rid].append(r)

            # iteramos sobre cada ID
            for rid, group in results_by_id.items():

                # buscamos solamente cajas-imagen
                box_res = None
                for item in group:
                    if item.get("type") in ["rectanglelabels", "polygonlabels"]:
                        box_res = item
                        break

                if box_res:
                    new_results.append(box_res)
                else:
                    final_text_list = resolve_text_for_group(group, full_text)

                    if len(final_text_list) > 1:
                        # si tenemos más de una corrección, avisamos
                        label_res = next(
                            (
                                r
                                for r in group
                                if r.get("type") in ["labels", "hypertextlabels"]
                            ),
                            None,
                        )
                        t = label_res.get("value", {}).get("text")
                        original_text = " ".join(t) if isinstance(t, list) else t
                        print(
                            f"En la tarea {task["id"]}, el fragmento \n\t > {original_text.replace("\n", "\n\t")}\n tiene múltiples correcciones:"
                        )
                        for correction in final_text_list:
                            print(f"\t > {correction.replace("\n", "\n\t")}")

                    final_text = final_text_list[
                        0
                    ]  # en el caso en el que haya más de una, cogemos la primera (placeholder)

                    if final_text and final_text.strip():
                        synthetic_res = {
                            "id": rid,
                            "type": "textarea",
                            "value": {"text": [final_text]},
                            "from_name": "text_adapter",
                            "to_name": "image",
                        }
                        new_results.append(synthetic_res)
                    else:
                        empty_boxes_on_page += 1

            # añadimos las relaciones
            new_results.extend(relations)

            # reemplazamos los resultados de las anotaciones
            ann["result"] = new_results

            if empty_boxes_on_page > 0:
                # imprimimos el número de cajas vacías que hemos encontrado

                print(
                    f"Tarea {task.get('id')}: Se han ignorado {empty_boxes_on_page} fragmentos de texto vacíos (o solo con espacios)."
                )

        # guardamos la tarea
        processed_tasks.append(new_task)

    # guardamos
    simplified_filepath.write_text(
        json.dumps(processed_tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Simplificadas {len(processed_tasks)} tareas.")
    print(f"Se han guardado en {simplified_filepath}")


def load_simplified_export(paths: PathBundle):
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
        return data

    if isinstance(data, dict):  # buscamos los diferentes resultados
        for k in ("tasks", "data", "results"):
            if isinstance(data.get(k), list):
                return data[k]

    raise ValueError(
        "El tipo de archivo que se ha especificado no tiene la estructura esperada."
    )
