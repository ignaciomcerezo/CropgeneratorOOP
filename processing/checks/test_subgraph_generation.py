from processing.augment_data.sequential.helpers import generate_connected_subgraphs

from preprocessing.tests.test_AnnotatedPage import Annotated_task_5

graph = Annotated_task_5.graph
ccs = Annotated_task_5.ordered_connected_components

for subgraph in generate_connected_subgraphs(
    Annotated_task_5.graph.keys(), Annotated_task_5.graph, 1
):
    pass
    # print(subgraph)

# TODO add proper tests to this all


subgraphs_generated = lambda k: set(
    [x for x in generate_connected_subgraphs(graph.keys(), graph, 1)]
)

sko1 = subgraphs_generated(1)

sko1_prime = set()

for fs in sko1:
    for x in fs:
        sko1_prime.add(x)

subgraphs_known_order_1 = set([x for x in graph.keys()])

assert set(sko1_prime) == set(
    subgraphs_known_order_1
), "Hay diferencia entre los subgrafos generados de orden 1 y los reales."
