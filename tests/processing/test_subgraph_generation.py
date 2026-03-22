from processing.sequential.helpers import generate_connected_subgraphs
from tests.tests_helper import load_particular_annotation

Annotated_task_5 = load_particular_annotation(5)

graph = Annotated_task_5.graph

for subgraph in generate_connected_subgraphs(
    Annotated_task_5.graph.keys(), Annotated_task_5.graph, 1
):
    pass

# TODO add proper preprocessing to this all


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
