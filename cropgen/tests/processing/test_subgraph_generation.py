from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.processing.sequential.helpers import generate_connected_subgraphs
from cropgen.shared.PathBundle import PathBundle
from cropgen.tests.tests_helper import load_particular_annotation


def _single_test_subgraph_generation(
    paths: PathBundle, lsi: LabelStudioInterface, ann: AnnotatedPage
):

    graph = ann.graph

    def subgraphs_generated(k) -> list[frozenset[str]]:
        return [
            subgraph
            for subgraph in generate_connected_subgraphs(graph.keys(), graph, k)
        ]

    sko1 = subgraphs_generated(1)

    sko1_prime = set()

    for fs in sko1:
        for x in fs:
            sko1_prime.add(x)

    subgraphs_known_order_1 = set([x for x in graph.keys()])

    assert set(sko1_prime) == set(
        subgraphs_known_order_1
    ), "Hay diferencia entre los subgrafos generados de orden 1 y los reales."


def test_subgraph_generation(paths, lsi, task_macedonia):
    for task_id in task_macedonia:
        ann = load_particular_annotation(paths, task_id)
        _single_test_subgraph_generation(paths, lsi, ann)
