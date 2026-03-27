def generate_connected_subgraphs(nodes, adj, k):
    """
    Genera todos los subgrafos conexos de orden k a partir del conjunto de
    nodos dado y el grafo de adyacencia.
    Devuelve un generador de conjuntos inmutables (frozenset) de nodos, lo que
    nos permite hacer un hash de toda la secuencia de nodos para evitar repeticiones
    más adelante.
    """
    sorted_nodes = sorted(list(nodes))

    # algoritmo de backtracking estándar, pero dentro de un objeto generador para
    # ser más eficiente (generar el grafo completo es expoonencial, así podemos
    # establecer un tiempo límite, y lo haremos)
    def backtrack(current_set, candidates):

        if len(current_set) == k:
            # si es de la longitud que buscamos, lo devolvemos (como generador que es)
            yield frozenset(current_set)
            return

        sorted_candidates = sorted(list(candidates))

        for i, node in enumerate(sorted_candidates):

            remaining = set(
                sorted_candidates[i + 1 :]
            )  # extensiones de subgrafo no exploradas

            new_extensions = set()  # posibles extensiones a partir de la actual

            for n in adj[node]:
                if n not in current_set and n not in candidates:
                    new_extensions.add(
                        n
                    )  # posible nueva extensión (no visitado y no en la actual)

            # coge el yield de una subllamada a sí mismo, completando
            yield from backtrack(
                current_set.union({node}), remaining.union(new_extensions)
            )

    # llamamos a la función inicial sobre todos los vértices del grafo.
    for start_node in sorted_nodes:
        valid_neighbors = {n for n in adj[start_node] if n > start_node}
        yield from backtrack({start_node}, valid_neighbors)
