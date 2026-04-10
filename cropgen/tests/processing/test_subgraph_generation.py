from cropgen.processing.AnnotatedPage import AnnotatedPage
from cropgen.tests.tests_helper import load_ann


def _single_test_subgraph_generation(ann: AnnotatedPage):
    for paragraph in ann.paragraphs:
        graph = set(ann.graph.keys())
        assert paragraph.subgraph is not None
        assert paragraph._subgraph_is_Pk()
        assert _is_subgraph(paragraph.subgraph, ann.graph)

        for order in range(len(paragraph)):

            for subsubgraph_keys in paragraph.generate_conntected_subgraphs(order):
                assert set(subsubgraph_keys).issubset(graph)


def test_subgraph_generation(paths, lsi, task_macedonia):
    for task_id in task_macedonia:
        ann = load_ann(paths, task_id, lsi=lsi, fake_image=True)
        _single_test_subgraph_generation(ann)


def _is_subgraph(subgraph: dict[str, set[str]], graph: dict[str, set[str]]) -> bool:
    assert set(subgraph.keys()).issubset(set(graph.keys()))

    for key in subgraph:
        if not subgraph[key].issubset(graph[key]):
            return False

    return True
