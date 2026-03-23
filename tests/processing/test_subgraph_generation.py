from processing.sequential.helpers import generate_connected_subgraphs
from tests.tests_helper import load_particular_annotation
from shared.PathBundle import PathBundle


def test_subgraph_generation():
    paths = PathBundle()
    ann5 = load_particular_annotation(paths, 5)

    graph = ann5.graph

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
