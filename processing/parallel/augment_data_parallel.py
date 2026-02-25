from multiprocessing import Pool, cpu_count
from paths import simplified_filepath
from processing.parallel.helpers import run_chunk, merge_excel_files
from parameters import orders_to_consider, generate_full_pages, max_samples_per_order
from downloaders.LabelStudioInterface import LabelStudioInterface
import numpy as np


def augment_data_parallel(
    orders_to_consider: list[int],
    generate_full_pages: bool,
    max_samples_per_order: int,
):

    def run_chunk_configured(chunk):
        return run_chunk(
            chunk,
            orders_to_consider=orders_to_consider,
            generate_full_pages=generate_full_pages,
            max_samples_per_order=max_samples_per_order,
            time_limit=10,
        )

    all_task_ids = [str(t.get("id")) for t in load_export(simplified_filepath)]

    # hacemos un shuffle para que no todas las cartas vayan al mismo proceso (son más sencillas)
    np.random.shuffle(all_task_ids)

    num_processes = max(1, int(cpu_count() * 0.8))

    # Calculamos el tamaño de cada tanda
    chunk_size = np.ceil(len(all_task_ids) / num_processes)
    chunks = []

    for i in range(num_processes):
        subset = all_task_ids[i * chunk_size : (i + 1) * chunk_size]

        if subset:
            chunks.append((subset, i))

    print(f"Procesando {len(all_task_ids)} tareas usando {num_processes} núcleos.")

    with Pool(
        processes=num_processes
    ) as pool:  # creamos tantos procesos como hayamos indicado

        # le adjudicamos a nuestros procesos las tareas (correr run_chunk en cada elemento de chunk)
        results = pool.map(run_chunk_configured, chunks)

    print("Procesado terminado, combinando excels...")

    # 6. Combine Results
    output_excel_name = "pairs.xlsx"
    merge_excel_files(base_name="pairs_part", output_name=output_excel_name)
