from collections import deque

from cropgen.processing.AnnotatedPage import AnnotatedPage
from PIL import Image


def _reconstruct_cycle(u, v, parent):
    path_u = []
    node = u
    while node is not None:
        path_u.append(node)
        node = parent.get(node)

    path_v = []
    node = v
    while node is not None:
        path_v.append(node)
        node = parent.get(node)

    ancestors_u = {node: idx for idx, node in enumerate(path_u)}
    lca_idx_v = next(idx for idx, node in enumerate(path_v) if node in ancestors_u)
    lca = path_v[lca_idx_v]
    lca_idx_u = ancestors_u[lca]

    u_to_lca = path_u[: lca_idx_u + 1]
    v_to_lca = path_v[: lca_idx_v + 1]
    return u_to_lca + list(reversed(v_to_lca[:-1]))


def minimum_cycle(graph):
    """Return the nodes of the shortest simple cycle in an undirected graph."""
    best_cycle = None

    for start in graph:
        dist = {start: 0}
        parent = {start: None}
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                    continue

                if parent.get(u) == v:
                    continue

                cycle = _reconstruct_cycle(u, v, parent)
                if best_cycle is None or len(cycle) < len(best_cycle):
                    best_cycle = cycle

    return best_cycle


def test_audit_graphs_are_Pks(lsi, paths):
    for task in lsi.simplified_tasks:
        for k_ann, ann in enumerate(task.annotations):

            # if ann.id < 600:
            #     continue

            path = paths.get_image_path_from_task(task)
            assert path is not None
            ann_obj = AnnotatedPage(
                ann,
                Image.open(path),
                usernames_labelstudio=lsi.usernames,
            )
            for paragraph in ann_obj.paragraphs:
                graph = paragraph.subgraph
                if not graph:
                    print(ann_obj)
                    print(f"Párrafo vacío: {repr(ann_obj)} | {paragraph}")
                    continue

                # Buscar nodos extremos (grado 1)
                ends = [n for n, v in graph.items() if len(v) == 1]
                if not ends:
                    # print(ann_obj)
                    # min_cycle = minimum_cycle(graph)
                    # print(
                    #     f"Párrafo sin extremos: {repr(ann_obj)} | {paragraph} | "
                    #     f"ciclo mínimo ({len(min_cycle) if min_cycle is not None else None}): {min_cycle}"
                    # )
                    continue

                start = ends[0]
                seen = {start}
                current = start
                prev = None
                is_path = True

                while True:
                    neighbors = [n for n in graph[current] if n != prev]
                    unvisited = [n for n in neighbors if n not in seen]
                    if len(unvisited) > 1:
                        is_path = False
                        break
                    if not unvisited:
                        break
                    next_node = unvisited[0]
                    if (
                        len(
                            [
                                n
                                for n in graph[next_node]
                                if n not in seen and n != current
                            ]
                        )
                        > 1
                    ):
                        is_path = False
                        break
                    seen.add(next_node)
                    prev, current = current, next_node

                if not is_path or len(seen) != len(graph):
                    min_cycle = minimum_cycle(graph)
                    print(f"{ann_obj}")
                    print(
                        f"Párrafo no isomorfo a un camino: {repr(ann_obj)} | {paragraph} | "
                        f"ciclo mínimo ({len(min_cycle) if min_cycle is not None else None}): {min_cycle}"
                    )
                    if min_cycle is not None:
                        img, trans, sindex = ann_obj.cluster_reading_order(min_cycle)
                        raise ValueError(
                            f"(Anotación {ann_obj.task_id}) - Detectado ciclo con transcripción {trans} y {sindex=}."
                        )
