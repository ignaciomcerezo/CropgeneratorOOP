from datasets import Dataset

from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.splitter.generation.generate_generator import (
    generate_generator,
    raw_features,
)


def get_datasets(
    pdi: PairsDataInterface,
    orders_to_split_with: list[int],
    p: float = 0.95,
    augment=True,
) -> tuple[Dataset, Dataset]:
    """
    A partir de las longitudes a considerar genera los dataframes con los datos y traduce
    la información de estos a instancias de Dataset
    """
    # creamos los datasets

    df_train, df_test = pdi.split(p, orders_to_split_with)

    generator_fn = generate_generator(pdi)
    print(f"Creando dataset de train ({augment=}...", end=" ")
    dataset_train = Dataset.from_generator(
        generator_fn,
        gen_kwargs={"df": df_train},
        features=raw_features,
    )
    print(f"Creadas {len(dataset_train)} muestras de entrenamiento")

    print("Creando datset de test (sin aumentar)...", end=" ")
    dataset_test = Dataset.from_generator(
        generator_fn,
        gen_kwargs={"df": df_test},
        features=raw_features,
    )
    print(f"Creadas {len(dataset_test)} muestras de test")

    return dataset_train, dataset_test
