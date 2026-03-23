from time import time
from shared.default_parameters import max_samples_per_order, time_limit_subgraph_generation


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


def create_reservoir(subgraph_gen, time_limit, max_samples):
    """
    Genera el reservorio de subgrafos a generar, aceptando o rechazando de manera
    probabilística cada posible subgrafo, hasta alcanzar el límite.
    """
    # si hemos escogido un max_samples_per_order, limitamos el generador
    # haciendo reservoir sampling
    reservoir = set()
    start_time = time.time()
    i = 0

    for seq_ids in subgraph_gen:

        if (time.time() - start_time) > time_limit_subgraph_generation:
            break

        # recordemos que hay repeticiones en la generación de subgrafos,
        # (A-B-C  y C-B-A se generan por separado, pero soon el
        # mismo grafo), de ahí el chequeo constante
        #      seq_ids not in reservoir
        # que hacemos a continuación, pues ni siquiera al principio
        # queremos llenar el reservorio con repeticiones. En caso
        # de hacerlo, podríamos terminar con repeticiones, pues
        # el reservorio original puede ser también con el que acabamos.

        if (i < max_samples_per_order) and (seq_ids not in reservoir):
            # al principio llenamos toda la reserva
            reservoir.add(seq_ids)
            i += 1

        elif seq_ids not in reservoir:

            # Reemplazamos aleatoriamente elementos en la reserva con probabilidad k/i
            j = np.random.randint(0, i)

            if j < max_samples_per_order:
                reservoir[j] = seq_ids

    return reservoir
