from pathlib import Path
from PIL import Image, ImageOps
from preprocessing.AnnotatedPage import AnnotatedPage
from tqdm.auto import tqdm
from processing.sequential.helpers import generate_connected_subgraphs
from preprocessing.helpers.helper_to_classes import (
    get_image_path_from_task,
    get_deterministic_id,
)
from parameters import (
    output_excel_name as default_output_excel_name,
    time_limit_subgraph_generation as default_time_limit_subgraph_generation,
)
from paths import (
    crops_path as default_crops_path,
    output_path as default_output_path,
)
import pandas as pd
from labelstudio.LabelStudioInterface import LabelStudioInterface


def augment_data_sequential(
    generate_full_pages: bool = True,
    generate_full_paragraphs: bool = True,
    pages_only: list[int] | None = None,
    tasks_only: list[int] | None = None,
    is_parallel: bool = False,
    output_excel_name: str = default_output_excel_name,
    additive_excel: bool = False,
    orders_to_consider: list[int] | str = "all",
    output_path: Path = default_output_path,
    crops_path: Path = default_crops_path,
    LSI: LabelStudioInterface | None = None,
):
    """Función principal para procesar las tareas y generar los recortes aumentados."""
    LSI = LSI if LSI else LabelStudioInterface()
    assert isinstance(output_path, Path)
    assert isinstance(crops_path, Path)
    output_path.mkdir(parents=True, exist_ok=True)
    crops_path.mkdir(parents=True, exist_ok=True)

    page_only = (
        [str(x).rjust(3, "0") for x in pages_only]
        if isinstance(pages_only, (list, tuple))
        else None
    )
    task_only = (
        [str(x) for x in tasks_only] if isinstance(tasks_only, (list, tuple)) else None
    )

    excel_filepath = Path(output_path) / output_excel_name

    tasks = LSI.simplified_tasks

    new_rows_data = []

    if excel_filepath.exists() and additive_excel:
        print(
            f"Archivo Excel existente detectado: {excel_filepath} (se añadirán los datos)"
        )
    else:
        print(f"Creando archivo excel: {excel_filepath}")

    if generate_full_pages:
        full_dir = crops_path / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
    if generate_full_paragraphs:
        paragraphs_dir = crops_path / "paragraphs"
        paragraphs_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    if (orders_to_consider == "all") or (orders_to_consider is None):
        orders_to_consider = None
    else:
        assert isinstance(
            orders_to_consider, list
        ), f'orders_to_consider debe ser una lista, NoneType, tupla o "all"'
        assert all(
            [isinstance(x, int) for x in orders_to_consider]
        ), "Si orders_to_consider viene dado como una lista, debe ser una lista de ints."

    # no se procesan todas las tareas?
    task_only_set = set(str(x) for x in task_only) if task_only else None
    page_only_set = set(str(x).rjust(3, "0") for x in page_only) if page_only else None
    filtering_active = (task_only_set is not None) or (page_only_set is not None)

    # total de tareas a procesar
    total_tqdm = len(tasks)
    total_tqdm = (
        min(total_tqdm, len(task_only)) if (task_only is not None) else total_tqdm
    )
    total_tqdm = (
        min(total_tqdm, len(page_only)) if (page_only is not None) else total_tqdm
    )

    progressbar = tqdm(total=total_tqdm)

    for task_idx, task in enumerate(tasks, start=1):
        task_id = str(task.get("id"))
        if is_parallel:
            # cuando paralelizamos, los splits se hacen por tareas.
            if task_id not in task_only_set:
                continue

        img_path = get_image_path_from_task(
            task
        )  # cogemos la imagen que le corresponde

        if img_path is None:
            print(f"No hay imagen para la tarea {task.get('id')}")
            continue

        page_number = img_path.stem if img_path else "N/A"

        if filtering_active:  # si no es paralelo y hay filtros (page_only o task_only)
            pageok = False  # pg. marcada para procesar
            taskok = False  # tarea marcada para procesar

            if page_only_set is not None:
                pageok = str(page_number) in page_only_set

            if task_only_set is not None:
                taskok = task_id in task_only_set

            if not (pageok or taskok):  # si no está en ningún filtro, no la procesamos
                continue

        progressbar.update(1)

        try:  # abrimos y preparamos la imagen
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)  # posible corrección de orientación
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            continue

        for Ann in (AnnotatedPage(ann, img, unrotate=False) for ann in LSI[task_id]):
            if generate_full_pages:
                image, transcription, sindex = Ann.cluster_reading_order(
                    list(Ann.graph.keys())
                )

                assert sindex == 0

                filename = f"pg_{page_number}_t{task_id}_h{get_deterministic_id(transcription)}.png"

                image.save(full_dir / filename)

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

                    if not paragraphs_dir.exists():
                        paragraphs_dir.mkdir()

                    image, transcription, sindex = Ann.cluster_reading_order(
                        paragraph.image_boxes_ids
                    )

                    filename = f"pg_{page_number}_t{task_id}_par{paragraph.index}_h{get_deterministic_id(transcription)}.png"
                    image.save(paragraphs_dir / filename)
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

                    order_dir = crops_path / str(order)

                    if not order_dir.exists():
                        order_dir.mkdir()

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
                        collage.save(order_dir / filename)

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

    # guardamos el excel con la correspondencia
    new_df = pd.DataFrame(new_rows_data)

    # comprobamos si hay que unirlo con un dataset anterior
    if excel_filepath.exists() and additive_excel:
        try:
            existing_df = pd.read_excel(excel_filepath)
            # concatenamos los dataframes
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error leyendo el archivo excel existente, lo sobreescribimos: {e}")
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

    # lo guardamos a un excel
    try:
        final_df.to_excel(excel_filepath, index=False)
        print(
            f"\nGenerados {total_saved} recortes aumentados y guardados en {output_excel_name}."
        )
    except Exception as e:
        print(f"Error guardando el archivo excel: {e}")
