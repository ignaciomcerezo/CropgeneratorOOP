from multiprocessing import Pool, cpu_count
from processing.parallel.helpers import run_chunk, merge_excel_files
import numpy as np
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from functools import partial
from shared.PathBundle import PathBundle


def augment_data_parallel(
    paths: PathBundle,
    orders_to_consider: list[int],
    generate_full_pages: bool,
    generate_paragraphs: bool,
    tasks_only: list[int] | None,
    lsi: LabelStudioInterface | None = None,
):
    lsi = lsi if (lsi is not None) else LabelStudioInterface(paths)

    simplified_tasks = lsi.simplified_tasks

    if tasks_only:
        tasks_only = [str(i) for i in tasks_only]
        simplified_tasks = [
            t for t in simplified_tasks if str(t.get("id")) in tasks_only
        ]

    all_task_ids = [str(t.get("id")) for t in simplified_tasks]

    # hacemos un shuffle para que no todas las cartas vayan al mismo proceso (son más sencillas)
    np.random.shuffle(all_task_ids)

    # no podemos usar una definición de función dentro de esta
    # porque luego multiprocessing no puede mandar esa información (cosas de implementación)
    run_chunk_configured = partial(
        run_chunk,
        paths=paths,
        orders_to_consider=orders_to_consider,
        generate_full_pages=generate_full_pages,
        generate_paragraphs=generate_paragraphs,
    )

    num_processes = min(len(all_task_ids), max(1, int(cpu_count() * 0.8)))

    # Calculamos el tamaño de cada tanda
    chunk_size = int(np.ceil(len(all_task_ids) / num_processes))
    chunks = []

    for i in range(num_processes):
        subset = all_task_ids[i * chunk_size : (i + 1) * chunk_size]

        if subset:
            chunks.append((subset, i))

    print(f"Procesando {len(all_task_ids)} tareas usando {num_processes} núcleos.")

    with Pool(processes=num_processes) as pool:
        # creamos tantos procesos como hayamos indicado

        # le adjudicamos a nuestros procesos las tareas (correr run_chunk en cada elemento de chunk)
        results = pool.map(run_chunk_configured, chunks)

    print("Procesado terminado, combinando excels...")

    # 6. Combine Results
    output_excel_name = "pairs.xlsx"
    merge_excel_files(
        base_name="pairs_part", output_name=output_excel_name, paths=paths
    )
