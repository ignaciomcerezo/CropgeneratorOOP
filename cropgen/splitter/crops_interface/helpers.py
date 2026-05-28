def greedy_page_split_df(
    df, p=0.8, orders: list[int] | tuple[int] = (1,)
) -> tuple[set[str], set[str]]:
    """
    Divide las páginas en dos grupos (que serán train y test), de forma que la relación
    #f(a)/(#f(a) + #f(b)) sea aproximadamente p, donde f(a) es el conjunto de archivos (muestras) en
    el grupo de páginas train.
    Emplea un algoritmo greedy, que no es óptimo, pero es suficientemente bueno (al fin y al cabo
    las particiones en 80-20 o cualquier otra cantidad son esencialmente arbitrarias). Emplea para la
    partición solamente las longitudes indicadas
    """

    df_p = df[df.order.isin(orders)]  # solamente los archivos que queremos considerar

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
