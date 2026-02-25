BIG_BOX_THRESHOLD = 0.4
min_nodes_for_big_box_removal = 5  # mínimo de nodos para considerar que el que existan grafos estrellados es un problema (si so menos, no se hace el big box check de grafos estrellados)

# parámetros para la generación
orders_to_consider = [1]
generate_full_pages = True
max_samples_per_order = 0  # dejar a 0 para que sean todas
time_limit_subgraph_generation = 1
output_excel_name = "pairs.xlsx"
page_only: list[int] = None
task_only: list[int] = None
time_limit = (10,)
is_parallel = (False,)
additive_excel = (False,)
