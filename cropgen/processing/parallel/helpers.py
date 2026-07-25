from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
import os
import re

import pandas as pd

from cropgen.processing.sequential.augment_data_sequential import (
    augment_data_sequential,
)
from cropgen.shared.PathBundle import PathBundle


def run_chunk(
    chunk_args,
    paths: PathBundle,
    orders_to_consider,
    generate_full_pages: bool,
    generate_paragraphs: bool,
    save_images: bool,
):
    """
    Función de aumento de datos para un solo bloque.
    """
    tasks_subset, worker_id = chunk_args

    # Cada proceso guarda los resultados a un fichero JSONL diferente.
    part_json_name = paths.get_worker_json_filepath(worker_id)
    augment_data_sequential(
        paths=paths,
        orders_to_consider=orders_to_consider,
        generate_full_pages=generate_full_pages,
        generate_full_paragraphs=generate_paragraphs,
        tasks_only=tasks_subset,
        in_parallel=True,
        worker_id=worker_id,
        save_images = save_images,
    )
    return f"Tarea del trabajador {worker_id} terminada."


def merge_jsonl_files(paths: PathBundle, delete_parts=True):
    """
    Combina los archivos json individuales en uno solo. Busca todos los ficheros que encajan con {base_name}_*.jsonl,
    los concatena y genera el archivo completo.
    """
    base_name = paths.json_filepath.stem
    extension = paths.json_filepath.suffix
    output_name = paths.json_filepath

    files_to_merge = []

    # buscamos los archivos tipo jsonl que coincidan con la estructura que buscamos
    for filename in os.listdir(paths.data_out_path):
        if re.match(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$", filename):
            files_to_merge.append(paths.data_out_path / filename)

    if not files_to_merge:
        raise FileNotFoundError(
            "No hay archivos JSON para mezclar de la forma especificada."
        )

    print(f"Combinando {len(files_to_merge)} archivos {extension.upper()}...")

    dfs = []
    for filepath in files_to_merge:
        try:
            dfs.append(pd.read_json(filepath, encoding="utf-8"))
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    try:
        json_data = combined_df.to_json(
            orient="records",
            force_ascii=False,
        )

        (paths.data_out_path / output_name).write_text(json_data, encoding="utf-8")
        print(
            f"Archivo {output_name.suffix.upper()} combinado guardado en {paths.data_out_path / output_name}"
        )
    except Exception as e:
        print(f"Error guardando el archivo combinado: {e}")

    # eliminamos los archivos originales
    if delete_parts:
        for f in files_to_merge:
            try:
                os.remove(f)
            except Exception as e:
                print(f"No se pudo eliminar {f}: {e}")
