import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.splitter.generation.get_dataset import get_datasets


def generate(
    paths: PathBundle | None = None,
    obi: OracleBucketInterface | None = None,
    lsi: LabelStudioInterface | None = None,
):
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
    load_dotenv()
    paths: PathBundle = PathBundle(Path(os.getcwd()).parent) if paths is None else paths

    obi: OracleBucketInterface = (
        OracleBucketInterface.from_env(paths) if obi is None else obi
    )
    obi.update()
    has_updated = LabelStudioInterface.update_conditional(paths)
    lsi: LabelStudioInterface = LabelStudioInterface(paths) if lsi is None else lsi
    if has_updated:
        lsi.save_simplified_export()

    augment_data_parallel(
        paths,
        tasks_only=None,
        orders_to_consider=[1, 2, 3],
        generate_full_pages=True,
        generate_paragraphs=True,
        lsi=lsi,
    )


def convert(paths: PathBundle) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    pdi = PairsDataInterface(paths)
    dataset_train, dataset_test = get_datasets(pdi, [1])

    n_samples = 100  # número de muestras
    np.random.seed(42)

    # seleccionamos los índices aleatorios
    samples_index = np.random.choice(
        len(dataset_test), n_samples, replace=False
    ).tolist()
    evals_index = np.random.choice(len(dataset_test), n_samples, replace=False).tolist()

    samples_test = dataset_test.select(samples_index)
    samples_eval = dataset_test.select(samples_index)

    return dataset_train, dataset_test, samples_test, samples_eval


def upload(
    dataset_train: Dataset,
    dataset_test: Dataset,
    samples_test: Dataset,
    samples_eval: Dataset,
):

    hub_name = os.environ["HUB_NAME"]
    complete_dataset = DatasetDict(
        {
            "train": dataset_train,
            "test": dataset_test,
            "samples_eval": samples_eval,
            "sample_test": samples_test,
        }
    )

    complete_dataset.push_to_hub(hub_name, private=True, token=True)

    print(f"Subido a https://huggingface.co/datasets/{hub_name}")


if __name__ == "__main__":
    generate()
    dataset_train, dataset_test, samples_test, samples_eval = convert()
    upload()
