from cropgen.processing.sequential.augment_data_sequential_new import (
    augment_data_sequential,
)
from cropgen.shared.PathBundle import PathBundle
from cropgen.shared.default_parameters import (
    orders_to_consider as default_orders_to_consider,
)
import pandas as pd
import os
import re


def run_chunk(
    chunk_args,
    paths: PathBundle,
    orders_to_consider=default_orders_to_consider,
    generate_full_pages=True,
    generate_paragraphs=True,
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
        is_parallel=True,
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
    for filename in os.listdir(paths.output_path):
        if re.match(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$", filename):
            files_to_merge.append(paths.output_path / filename)

    if not files_to_merge:
        raise FileNotFoundError(
            "No hay archivos JSON para mezclar de la forma especificada."
        )

    print(f"Combinando {len(files_to_merge)} archivos {extension.upper()}...")

    dfs = []
    for filepath in files_to_merge:
        try:
            dfs.append(pd.read_json(filepath, lines=True))
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    try:
        combined_df.to_json(
            paths.output_path / output_name,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print(
            f"Archivo {output_name.suffix.upper()} combinado guardado en {paths.output_path / output_name}"
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
