import numpy as np


def montecarlo_page_split_df(
    df, p=0.95, orders: list[int] | tuple[int] = (1,), n_trials: int = 1000
) -> tuple[set[str], set[str]]:
    """
    Emplea el método de monte carlo para dividir las páginas en train y test,
    empleando n_trials como máximo de intentos
    """

    print(f"Performing Montecarlo page split with {n_trials} trials")

    df_p = df[df.order.isin(orders)]

    page_counts = df_p.groupby("page").size()
    pages = page_counts.index.to_numpy()
    counts = page_counts.to_numpy()

    total_lines = counts.sum()
    target_a = total_lines * p

    best_error = float("inf")
    best_split_idx = 0
    best_pages = pages

    # usamos ahora un método de monte carlo para encontrar una partición válida
    # Método de Monte Carlo para encontrar la permutación con el mejor punto de corte
    for _ in range(n_trials):
        # creamos un orden aleatorio de las páginas.
        idx = np.random.permutation(len(pages))
        shuffled_pages = pages[idx]
        shuffled_counts = counts[idx]

        # calculamos la proporción de líneas que iría si cortamos en cada índice
        cum_lines = np.cumsum(shuffled_counts)

        # encontramos el índice que se asemeja más a p
        errors = np.abs(cum_lines - target_a)
        min_err_idx = np.argmin(errors)

        if errors[min_err_idx] < best_error:
            best_error = errors[min_err_idx]
            best_split_idx = min_err_idx
            best_pages = shuffled_pages

    # dividimos usando el mejor índice
    a = set(best_pages[: best_split_idx + 1])
    b = set(best_pages[best_split_idx + 1 :])

    return a, b
