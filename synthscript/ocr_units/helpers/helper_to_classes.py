def get_connected_components(adj: dict[str, set]):
    """
    Dado un grafo de adyacencia, devuelve las componentes conexas como una lista
    de conjuntos de nodos.
    """
    # backtracking habitual no recursivo para generar las componentes conexas de
    # un grafo usando un diccionario
    visited = set()
    components = []

    for v in adj:
        if v not in visited:  # si es la primera vez que vemos este nodo,
            comp = set()
            q = [v]
            while q:
                curr = q.pop(0)
                if curr in visited:
                    continue
                # añadimos el nodo a visitados y a la componente actual
                visited.add(curr)
                comp.add(curr)
                # añadimos los nodos adyacentes al actual a la lista para procesar
                # pues deben estar en la misma componente conexa.
                q.extend(list(adj.get(curr, [])))
            # añadimos la componente conexa
            components.append(comp)
    return components


def subdictionary(nodes, adj) -> dict[str, set[str]]:
    subdict = {}
    for node in nodes:
        subdict[node] = adj[node]
    return subdict


def is_path_graph(graph_dict):
    """
    checks if a graph is isomorphic to a path graph by checking if it is connected and
    its degree sequence matches that of a path graph.
    """
    n = len(graph_dict)

    if n == 0:
        return False
    if n == 1:
        return len(list(graph_dict.values())[0]) == 0

    degrees = [len(neighbors) for neighbors in graph_dict.values()]

    if degrees.count(1) != 2 or degrees.count(2) != n - 2:
        return False

    visited = set()

    start_node = next(
        node for node, neighbors in graph_dict.items() if len(neighbors) == 1
    )

    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph_dict[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return len(visited) == n
