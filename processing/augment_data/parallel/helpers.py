from processing.checks.augment_data_sequential import (
    augment_data_sequential,
)
from paths import simplified_filepath, output_path
import pandas as pd
import os
import re


def run_chunk(
    chunk_args,
    orders_to_consider=[1, 2, 3],
    generate_full_pages=True,
    max_samples_per_order=0,
    time_limit_subgraph_generation=0,
):
    """
    Función de aumento de datos para un solo bloque.
    """
    tasks_subset, worker_id = chunk_args

    # Cada proceso guarda los resultados a un excel diferente.
    part_excel_name = f"pairs_part_{worker_id}.xlsx"

    augment_data_sequential(
        simplified_filepath=simplified_filepath,
        output_excel_name=part_excel_name,
        orders_to_consider=orders_to_consider,
        generate_full_pages=generate_full_pages,
        task_only=tasks_subset,
        max_samples_per_order=max_samples_per_order,
        time_limit_subgraph_generation=time_limit_subgraph_generation,
        is_parallel=True,
    )
    return f"Tarea del trabajador {worker_id} terminada."


def merge_excel_files(base_name, output_name, delete_parts=True):
    """
    Combina los archivos excel individuales en uno solo.
    Finds all files matching base_name_*.xlsx, combines them,
    and saves to output_name.
    """

    files_to_merge = []

    # buscamos los archivos tipo xlsx que coincidan con la estructrua que buscamos
    for filename in os.listdir(output_path):
        if re.match(rf"^{re.escape(base_name)}_(\d+)\.xlsx$", filename):
            files_to_merge.append(output_path / filename)

    if not files_to_merge:
        print("No hay archivos de la forma especificada.")
        return

    print(f"Combinando {len(files_to_merge)} archivos...")

    dfs = []
    for filepath in files_to_merge:
        try:
            dfs.append(pd.read_excel(filepath))
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_excel(output_path / output_name, index=False)
    print(f"Archivo excel combinado guardado en {output_path / output_name}")

    # eliminamos los archivos originales
    if delete_parts:
        for f in files_to_merge:
            os.remove(f)
