from labelstudio.simplify_export import load_simplified_export
from pathlib import Path
from PIL import Image, ImageOps
from preprocessing.AnnotatedPage import AnnotatedPage
from tqdm.auto import tqdm
from processing.augment_data.sequential.helpers import (
    generate_connected_subgraphs,
    create_reservoir,
)
from preprocessing.helpers.helper_to_classes import (
    get_image_path_from_task,
    get_deterministic_id,
)
from paths import crops_path as default_crops_path, output_path as default_output_path
import pandas as pd


def augment_data_sequential(
    simplified_filepath,
    output_excel_name="pairs.xlsx",
    orders_to_consider="all",
    generate_full_pages=True,
    page_only: list[int] = None,
    task_only: list[int] = None,
    max_samples_per_order=0,
    time_limit_subgraph_generation=10,
    is_parallel=False,
    additive_excel=False,
    output_path: Path = default_output_path,
    crops_path: Path = default_crops_path,
):
    """
    Función principal para procesar las tareas y generar los recortes aumentados.
    """
    output_path = output_path if isinstance(output_path, Path) else Path(output_path)
    crops_path = crops_path if isinstance(crops_path, Path) else Path(crops_path)

    # nos aseguramos de que las carpetas de salida existen
    output_path.mkdir(parents=True, exist_ok=True)
    crops_path.mkdir(parents=True, exist_ok=True)

    # filtros por si solamente queremos procesar unas páginas o tareas concretas
    page_only = (
        [str(x).rjust(3, "0") for x in page_only]
        if isinstance(page_only, (list, tuple))
        else None
    )
    task_only = (
        [str(x) for x in task_only] if isinstance(task_only, (list, tuple)) else None
    )

    if not isinstance(simplified_filepath, Path):
        simplified_filepath = Path(simplified_filepath)

    if not simplified_filepath.exists():
        print(
            f"No existe el archivo de datos exportados simplificado de LabelStudio (indicado {simplified_filepath})"
        )
        return

    tasks = load_simplified_export(simplified_filepath)

    excel_path = Path(output_path) / output_excel_name

    new_rows_data = []

    # la creación del excel se hace al final.
    if excel_path.exists() and additive_excel:
        print(
            f"Archivo Excel existente detectado: {excel_path} (se añadirán los datos)"
        )
    else:
        print(f"Creando archivo excel: {excel_path}")

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

        annotations = task.get("annotations", [])

        # por cada anotación
        for Ann in (
            AnnotatedPage(ann, img, unrotate=False, cc_ordering=True)
            for ann in annotations
        ):
            if not Ann.image_boxes:
                continue  # si la anotación está vacía, simplemente pasamos

            # ---------------------------------------------------------
            # GENERACIÓN PÁGINA COMPLETA
            # ---------------------------------------------------------
            if generate_full_pages:
                FULL_DIR = crops_path / "full"
                FULL_DIR.mkdir(parents=True, exist_ok=True)
                full_filename = f"pg_{page_number}_t{task_id}_full.png"

                # Obtenemos todos los IDs de la página para pasarlos a la función
                all_boxes_sequence = list(Ann.image_boxes.keys())

                try:
                    full_page_collage, full_page_transcription, sindex = (
                        Ann.cluster_reading_order(list(Ann.graph.keys()))
                    )

                    full_page_collage.save(FULL_DIR / full_filename)

                    # sanity check
                    assert (
                        sindex == 0
                    ), f"La concatenación de todas las cajas en la tarea {task_id} no consigue un sindex de 0 en Ann.concatenate_transcriptions(...)."

                    new_rows_data.append(
                        {  # nueva fila para el dataframe
                            "task": task_id,
                            "order": "FULL",
                            "sindex": 0,
                            "text": full_page_transcription,
                            "page": page_number,
                            "crop_file": full_filename,
                            "background_color": Ann.background_color,
                        }
                    )
                    total_saved += 1
                except Exception as e:
                    print(f"Error guardando la página completa {full_filename}. {e}")

            # ---------------------------------------------------------
            # GENERACIÓN SUBGRAFOS
            # ---------------------------------------------------------

            # esto ya se hace dentro de la anotación
            # Ann.trim_star_nodes(BIG_BOX_THRESHOLD, min_nodes_for_big_box_removal)

            ccomponents = Ann.ordered_connected_components

            max_comp_size = max(len(c) for c in ccomponents) if ccomponents else 0
            saved_subgraphs_ids = set()  # para evitar duplicados

            range_orders = (
                [a for a in orders_to_consider if a <= max_comp_size]
                if (orders_to_consider is not None)
                else range(1, max_comp_size + 1)
            )

            for order in range_orders:
                # carpeta de outputs para esta longitud
                order_dir = crops_path / str(order)

                # buscamos subgrafos en componentes conexas de tamaño suficiente
                big_enough_cc = [c for c in ccomponents if len(c) >= order]

                for idx_comp, comp in enumerate(
                    big_enough_cc
                ):  # por cada componente conexa
                    subgraph_gen = generate_connected_subgraphs(comp, Ann.graph, order)

                    if max_samples_per_order > 0:
                        reservoir = create_reservoir(
                            subgraph_gen,
                            time_limit_subgraph_generation,
                            max_samples_per_order,
                        )
                    else:
                        reservoir = subgraph_gen

                    order_dir.mkdir(exist_ok=True)

                    for box_id_sequence in reservoir:

                        seq_hash = get_deterministic_id(
                            "".join(sorted(box_id_sequence))
                        )[:16]
                        filename = (
                            f"pg_{page_number}_t{task_id}_order{order}_h{seq_hash}.png"
                        )

                        if seq_hash in saved_subgraphs_ids:
                            # esto debería ser imposible a no ser que no haya reservoir.
                            continue

                        try:
                            collage, transcripcion, sindex = Ann.cluster_reading_order(
                                box_id_sequence
                            )
                            collage.save(order_dir / filename)

                            new_rows_data.append(
                                {  # nueva fila para el dataframe
                                    "task": task_id,
                                    "order": order,
                                    "text": transcripcion,
                                    "page": page_number,
                                    "crop_file": filename,
                                    "sindex": sindex,
                                    "background_color": Ann.background_color,
                                }
                            )

                            total_saved += 1
                            saved_subgraphs_ids.add(seq_hash)
                        except Exception as e:
                            print(f"Error saving {filename} file. {e}")

    # guardamos el excel con la correspondencia
    new_df = pd.DataFrame(new_rows_data)

    # comprobamos si hay que unirlo con un dataset anterior
    if excel_path.exists() and additive_excel:
        try:
            existing_df = pd.read_excel(excel_path)
            # concatenamos los dataframes
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error leyendo el archivo excel existente, lo sobreescribimos: {e}")
            final_df = new_df
    else:
        if new_df.empty:
            final_df = pd.DataFrame(columns=["task", "order", "crop_file", "text"])
        else:
            final_df = new_df

    # lo guardamos a un excel
    try:
        final_df.to_excel(excel_path, index=False)
        print(
            f"\nGenerados {total_saved} recortes aumentados y guardados en {output_excel_name}."
        )
    except Exception as e:
        print(f"Error guardando el archivo excel: {e}")
