def greedy_page_split_df(df, p=0.8, lengths: tuple[int] = (1,)):
    """
    Divide las páginas en dos grupos (que serán train y test), de forma que la relación
    #f(a)/(#f(a) + #f(b)) sea aproximadamente p, donde f(a) es el conjunto de archivos (muestras) en
    el grupo de páginas train.
    Emplea un algoritmo greedy, que no es óptimo, pero es suficientemente bueno (al fin y al cabo
    las particiones en 80-20 o cualquier otra cantidad son esencialmente arbitrarias). Emplea para la
    partición solamente las longitudes indicadas
    """

    df_p = df[df.order.isin(lengths)]  # solamente los archivos que queremos considerar

    total = df_p.count().iloc[0]

    target_cardfa = int(total * p)  # número de archivos buscado

    a = []
    b = []

    fa_card = 0

    count_boxes = lambda page: len(df_p[df_p["page"] == page])

    pageandfilecount = [(page, count_boxes(page)) for page in df_p["page"].unique()]

    for page, file_count in sorted(pageandfilecount, key=lambda x: x[1]):

        # comprueba si añadir la página nos acerca o nos aleja del objetivo

        diff_if_add = abs((fa_card + file_count) - target_cardfa)
        diff_if_skip = abs(fa_card - target_cardfa)

        if diff_if_add < diff_if_skip:
            # si nos acerca, la metemos en A
            a.append(page)
            fa_card += file_count
        else:
            # si nos aleja, la metemos en B
            b.append(page)

    return set(a), set(b)


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

    a = train_pages_laloma.union(train_pages_letters)
    b = test_pages_laloma.union(test_pages_laloma)

    train = df[df["page"].isin(a)]
    test = df[df["page"].isin(b)]

    print(f"Split total de {len(train)/(len(train)+len(test))}")

    return train, test
