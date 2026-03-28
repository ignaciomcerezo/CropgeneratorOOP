from pathlib import Path
from PIL import Image, ImageOps
from cropgen.processing.AnnotatedPage import AnnotatedPage
from tqdm.auto import tqdm
from cropgen.processing.sequential.helpers import generate_connected_subgraphs
from cropgen.processing.helpers.helper_to_classes import (
    get_deterministic_id,
)
from cropgen.shared.PathBundle import (
    PathBundle,
    _output_json_filename as default_json_name,
)
import pandas as pd
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface


def augment_data_sequential(
    paths: PathBundle,
    generate_full_pages: bool = True,
    generate_full_paragraphs: bool = True,
    tasks_only: list[int] | None = None,
    is_parallel: bool = False,
    output_json_name: str = default_json_name,
    additive_json: bool = False,
    orders_to_consider: list[int] | str = "all",
    lsi: LabelStudioInterface | None = None,
):
    """Función principal para procesar las tareas y generar los recortes aumentados."""
    lsi = lsi if lsi else LabelStudioInterface(paths)
    paths.output_path.mkdir(parents=True, exist_ok=True)
    paths.crops_path.mkdir(parents=True, exist_ok=True)

    task_only = (
        [str(x) for x in tasks_only] if isinstance(tasks_only, (list, tuple)) else None
    )

    jsonl_filepath = Path(paths.output_path) / output_json_name

    tasks = lsi.simplified_tasks

    new_rows_data = []

    if jsonl_filepath.exists() and additive_json:
        print(
            f"Archivo JSONL existente detectado: {jsonl_filepath} (se añadirán los datos)"
        )
    else:
        print(f"Creando archivo jsonl: {jsonl_filepath}")

    total_saved = 0

    task_only_set, filtering_active, progressbar = _process_orders_to_consider(
        orders_to_consider, task_only, len(tasks)
    )

    for task_idx, task in enumerate(tasks, start=1):
        task_id = str(task.get("id"))
        if is_parallel:
            # cuando paralelizamos, los splits se hacen por tareas.
            if filtering_active and (task_id not in task_only_set):
                continue

        img_path = paths.get_image_path_from_task(
            task
        )  # cogemos la imagen que le corresponde

        if img_path is None:
            print(f"No hay imagen para la tarea {task.get('id')}")
            continue

        page_number = img_path.stem if img_path else "N/A"

        if (not is_parallel) and filtering_active:
            # Solo filtrar por task_only_set
            if task_id not in task_only_set:
                continue

        progressbar.update(1)

        try:  # abrimos y preparamos la imagen
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)  # posible corrección de orientación
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            continue

        for Ann in (
            AnnotatedPage(ann, img, unrotate=False, usernames_LS=lsi.usernames)
            for ann in lsi[task_id]
        ):
            if generate_full_pages:
                full_dir = paths.get_order_folder("full")

                image, transcription, sindex = Ann.cluster_reading_order(
                    list(Ann.graph.keys())
                )

                assert sindex == 0

                filename = f"pg_{page_number}_t{task_id}_h{get_deterministic_id(transcription)}.png"

                filepath = full_dir / filename
                image.save(filepath)

                new_rows_data.append(
                    {  # nueva fila para el dataframe
                        "task": task_id,
                        "paragraph": "full",
                        "order": "full",
                        "sindex": 0,
                        "text": transcription,
                        "page": page_number,
                        "crop_file": filename,
                        "background_color": Ann.background_color,
                        "average_rotation": Ann.get_average_rotation(Ann.graph.keys()),
                    }
                )
                total_saved += 1

            for paragraph in Ann.paragraphs:
                if generate_full_paragraphs and not (
                    Ann.is_single_paragraph and generate_full_pages
                ):
                    paragraph_dir = paths.get_order_folder("paragraph")

                    image, transcription, sindex = Ann.cluster_reading_order(
                        paragraph.image_boxes_ids
                    )

                    filename = f"pg_{page_number}_t{task_id}_par{paragraph.index}_h{get_deterministic_id(transcription)}.png"

                    filepath = paragraph_dir / filename
                    image.save(filepath)

                    new_rows_data.append(
                        {  # nueva fila para el dataframe
                            "task": task_id,
                            "order": "paragraph",
                            "paragraph": paragraph.index,
                            "sindex": sindex,
                            "text": transcription,
                            "page": page_number,
                            "crop_file": filename,
                            "background_color": Ann.background_color,
                            "average_rotation": Ann.get_average_rotation(
                                paragraph.subgraph.keys()
                            ),
                        }
                    )
                    total_saved += 1

                saved_subgraphs_ids = set()  # para evitar duplicados

                for order in range(
                    1, len(paragraph) - generate_full_paragraphs + 1
                ):  # aquí ya forzamos que no se generen dos veces los párrafos completos. Sin embargo si
                    # generate_full_paragraphs = False, sí que los generamos si cumplen el orden (lo que no hacemos
                    # es repetir generación).

                    if order not in orders_to_consider:
                        continue

                    for box_id_sequence in generate_connected_subgraphs(
                        paragraph.image_boxes_ids, paragraph.subgraph, order
                    ):
                        seq_hash = get_deterministic_id(
                            "".join(sorted(box_id_sequence))
                        )

                        if seq_hash in saved_subgraphs_ids:
                            # esto debería ser imposible a no ser que no haya reservoir.
                            continue

                        filename = f"pg_{page_number}_t{task_id}_par{paragraph.index}_order{order}_h{seq_hash}.png"

                        collage, transcripcion, sindex = Ann.cluster_reading_order(
                            box_id_sequence
                        )

                        filepath = paths.get_order_folder(order) / filename
                        collage.save(filepath)

                        new_rows_data.append(
                            {  # nueva fila para el dataframe
                                "task": task_id,
                                "order": order,
                                "paragraph": paragraph.index,
                                "sindex": sindex,
                                "text": transcripcion,
                                "page": page_number,
                                "crop_file": filename,
                                "background_color": Ann.background_color,
                                "average_rotation": Ann.get_average_rotation(
                                    box_id_sequence
                                ),
                            }
                        )

                        total_saved += 1
                        saved_subgraphs_ids.add(seq_hash)

    # guardamos en JSONL con la correspondencia
    new_df = pd.DataFrame(new_rows_data)

    # comprobamos si hay que unirlo con un dataset anterior (JSONL)
    if jsonl_filepath.exists() and additive_json:
        try:
            existing_df = pd.read_json(jsonl_filepath, lines=True)
            # concatenamos los dataframes
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error leyendo el archivo jsonl existente, lo sobreescribimos: {e}")
            final_df = new_df
    else:
        if new_df.empty:
            final_df = pd.DataFrame(
                columns=[
                    "task",
                    "page",
                    "order",
                    "paragraph",
                    "sindex",
                    "text",
                    "crop_file",
                    "background_color",
                    "average_rotation",
                ]
            )
        else:
            final_df = new_df

    # lo guardamos a un JSONL (one-record-per-line)
    try:
        final_df.to_json(
            jsonl_filepath, orient="records", lines=True, force_ascii=False
        )
        print(
            f"\nGenerados {total_saved} recortes aumentados y guardados en {output_json_name}."
        )
    except Exception as e:
        print(f"Error guardando el archivo jsonl: {e}")


def _process_orders_to_consider(
    orders_to_consider: list[int] | str,
    task_only: list[str] | None,
    len_tasks: int,
):
    if (orders_to_consider == "all") or (orders_to_consider is None):
        orders_to_consider = None
    else:
        assert isinstance(
            orders_to_consider, list
        ), 'orders_to_consider debe ser una lista, NoneType, tupla o "all"'
        assert all(
            [isinstance(x, int) for x in orders_to_consider]
        ), "Si orders_to_consider viene dado como una lista, debe ser una lista de ints."

    # Filtrado solo por tasks
    task_only_set = set(str(x) for x in task_only) if task_only else None
    filtering_active = task_only_set is not None

    # total de tareas a procesar
    total_tqdm = len(task_only) if filtering_active else len_tasks
    progressbar = tqdm(total=total_tqdm)

    if not filtering_active:
        print(f"Procesando todas las tareas (sin filtro de tasks_only)")
    else:
        print(f"Filtrando solo las tareas con ids: {task_only}")

    return task_only_set, filtering_active, progressbar
