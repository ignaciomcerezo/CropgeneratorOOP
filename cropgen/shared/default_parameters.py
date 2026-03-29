big_box_threshold = 0.4
min_nodes_for_big_box_removal = 5  # mínimo de nodos para considerar que el que existan grafos estrellados es un problema (si so menos, no se hace el big box check de grafos estrellados)

# parámetros para la generación de recortes
orders_to_consider = [1]
generate_full_pages = True
max_samples_per_order = 0  # dejar a 0 para que sean todas
time_limit_subgraph_generation = 1
page_only: list[int] | None = None
task_only: list[int] | None = None
time_limit = 10
is_parallel = False
additive_json = False

# parámetros para la generación del dataset
min_context_chars = 50
context_chars = 100  # todo: implement this one now...
min_context_words = 10
context_words = 15
