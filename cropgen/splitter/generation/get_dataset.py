from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.splitter.generation.generate_generator import (
    generate_generator,
    raw_features,
)
from datasets import Dataset


def get_datasets(
    pdi: PairsDataInterface,
    orders_to_consider: list[int],
    p: float = 0.9,
    augment=True,
    resize_scale=1,
):
    """
    A partir de las longitudes a considerar genera los dataframes con los datos y traduce
    la información de estos a instancias de Dataset
    """
    # creamos los datasets

    df_train, df_test = pdi.split(p, orders_to_consider)

    print(f"Creando dataset de train ({augment=}...", end=" ")
    dataset_train = Dataset.from_generator(
        generate_generator(df_train, augment=augment, resize_scale=resize_scale),
        features=raw_features,
    )
    print(f"Creadas {len(dataset_train)} muestras de entrenamiento")

    print("Creando datset de test (sin aumentar)...", end=" ")
    dataset_test = Dataset.from_generator(
        generate_generator(df_test, resize_scale=resize_scale),
        features=raw_features,
    )
    print(f"Creadas {len(dataset_test)} muestras de test")

    return dataset_train, dataset_test
