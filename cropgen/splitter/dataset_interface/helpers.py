import re


def prev_page(page, pages):
    """
    Devuelve la página anterior a "page" si está en "pages" siguiendo el mismo formato que tienen los nombres
    de las páginas (uno para LaLoMa y otro para las cartas). Si no hay página previa, devuelve -1
    """

    if page.isdigit():  # si es una página de LaLoMa

        prev = str(int(page) - 1).rjust(3, "0")

        if prev in pages:
            return prev
        else:
            return False

    else:  # si no es una página de LaLoMa
        matched = re.search(r"_p(\d*).png$", page) or re.search(r"_p(\d*)$", page)

        if matched is None:
            raise ValueError(
                f"No tiene el formato de página esperado: {page=}, {matched=}"
            )

        subpage = int(
            matched.group(1)
        )  # las cartas tienen estructura nombre_p\d*.png, donde \d* son dígitos.
        if subpage > 1:
            w = page.replace(f"_p{subpage}", f"_p{subpage-1}")

            if w in pages:
                return w
            else:
                return False
        else:
            return False


def greedy_page_split_df(df, p=0.8, lengths: tuple[int] = (1,)):
    """
    Divide las páginas en dos grupos (que serán train y test), de forma que la relación
    #f(A)/(#f(A) + #f(B)) sea aproximadamente p, donde f(A) es el conjunto de archivos (muestras) en
    el grupo de páginas train.
    Emplea un algoritmo greedy, que no es óptimo, pero es suficientemente bueno (al fin y al cabo
    las particiones en 80-20 o cualquier otra cantidad son esencialmente arbitrarias). Emplea para la
    partición solamente las longitudes indicadas
    """

    df_p = df[df.order.isin(lengths)]  # solamente los archivos que queremos considerar

    total = df_p.count().iloc[0]

    target_fA_count = int(total * p)  # número de archivos buscado

    A = []
    B = []

    fA_count = 0

    count_boxes = lambda page: len(df_p[df_p["page"] == page])

    pageandfilecount = [(page, count_boxes(page)) for page in df_p["page"].unique()]

    for page, file_count in sorted(pageandfilecount, key=lambda x: x[1]):

        # comprueba si añadir la página nos acerca o nos aleja del objetivo

        diff_if_add = abs((fA_count + file_count) - target_fA_count)
        diff_if_skip = abs(fA_count - target_fA_count)

        if diff_if_add < diff_if_skip:
            # si nos acerca, la metemos en A
            A.append(page)
            fA_count += file_count
        else:
            # si nos aleja, la metemos en B
            B.append(page)

    return set(A), set(B)


def get_split_separate_laloma_and_letters(
    df, prop_train=0.8, orders: tuple[int] = (1,)
):
    """
    Divide los nombres de los archivos según longitudes en train y test usando greedy_page_split.
    Solamente tiene en cuenta para hacer la proporción las longitudes que se encuentren en "lengths"
    """

    train_pages_laloma, test_pages_laloma = greedy_page_split_df(
        df[~df["is_letter"]], prop_train, lengths=orders
    )

    train_pages_letters, test_pages_laloma = greedy_page_split_df(
        df[df["is_letter"]], prop_train, lengths=orders
    )

    # dividimos de forma homogénea el train y el test

    A = train_pages_laloma.union(train_pages_letters)
    B = test_pages_laloma.union(test_pages_laloma)

    train = df[df["page"].isin(A)]
    test = df[df["page"].isin(B)]

    print(f"Split total de {len(train)/(len(train)+len(test))}")

    return train, test
