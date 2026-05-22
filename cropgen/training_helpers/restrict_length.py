def restrict_length(dataset, acceptable_lengths=[1], transform_func=None):
    """Toma los elementos del dataset cuyo orden esté entre las especificadas"""

    if transform_func is None:
        raise ValueError(
            "Se debe dar una función de transformación, aunque sea lambda x: x."
        )

    acceptable_lengths = set([str(x) for x in acceptable_lengths])

    dataset.set_transform(lambda x: x)
    subdataset = dataset.filter(lambda x: str(x["order"]) in acceptable_lengths)

    if transform_func:
        subdataset.set_transform(transform_func)

    return subdataset
