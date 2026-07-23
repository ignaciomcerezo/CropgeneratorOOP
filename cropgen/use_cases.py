import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
import json
from huggingface_hub import HfApi

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from cropgen.splitter.generation.get_dataset import get_datasets


def setup(path: str | Path | None = None,
    obi: OracleBucketInterface | None = None,
    lsi: LabelStudioInterface | None = None,
    online: bool = True,
    project_id: int = 4
    ):
    """
    Descarga todos los archivos necesarios para crear el conjunto de datos, y genera sus respectivas interfaces.
    """
    load_dotenv()
    path: Path = Path(path) if path is not None else Path(os.getcwd()).parent
    paths = PathBundle(path)

    obi: OracleBucketInterface = (
        OracleBucketInterface.from_env(paths) if obi is None else obi
    )
    obi.update()

    lsi = LabelStudioInterface.from_env(paths, online, project_id) if lsi is None else lsi

    lsi.fetch_and_simplify()

    paths.lsi = lsi
    paths.obi = obi

    return obi, lsi
    



def generate(
    paths: PathBundle,
    orders_to_consider=[1, 2, 3],
    generate_full_pages=True,
    generate_paragraphs=True,
):
    """
    Genera los recortes.
    """
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1

    if paths.lsi is None:
        raise ValueError("Ejecuta setup(paths, ...) primero.")

    augment_data_parallel(
        paths,
        tasks_only=None,
        orders_to_consider=orders_to_consider,
        generate_full_pages=generate_full_pages,
        generate_paragraphs=generate_paragraphs,
        lsi=paths.lsi,
    )


def convert(
    paths: PathBundle,
    p: float = 0.95,
    orders_to_split_with: list[int] = [1],
    n_samples_eval=300,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Divide el conjunto de datos en train y test, y lo transforma ene un conjunto de datos de Huggingface
    """
    pdi = PairsDataInterface(paths)
    dataset_train, dataset_test = get_datasets(pdi, orders_to_split_with, p)

    np.random.seed(42)
    evals_index = np.random.choice(
        len(dataset_test), n_samples_eval, replace=False
    ).tolist()

    samples_eval = dataset_test.select(evals_index)

    return dataset_train, dataset_test, samples_eval


def upload(
    dataset_train: Dataset,
    dataset_test: Dataset,
    samples_eval: Dataset,
    hf_token: str | None = None,
    hub_name: str | None = None,
):
    if hub_name is None:
        if "HUB_NAME" not in os.environ:
            raise ValueError("Si no se pasa un hub_name, HUB_NAME debe ser una variable de entorno válida")
        hub_name: str = str(os.getenv("HUB_NAME"))
    if hf_token is None:
        if "HF_TOKEN" not in os.environ:
            raise ValueError("Si no se pasa un hf_token, HF_TOKEN debe ser una variable de entorno válida")
        hf_token: str = str(os.getenv("HF_TOKEN"))
    
    

    complete_dataset = DatasetDict(
        {
            "train": dataset_train,
            "test": dataset_test,
            "samples_eval": samples_eval,
        }
    )
    complete_dataset.push_to_hub(hub_name, private=True, token=hf_token)

    split_data = {
        "pages_train": sorted(list(set(dataset_train["page"]))),
        "pages_test": sorted(list(set(dataset_test["page"]))),
    }

    split_data_path = "page_splits_abcdefg.json"

    with open(split_data_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)

    api = HfApi(token= hf_token)
    api.upload_file(
        path_or_fileobj=split_data_path,
        path_in_repo="page_splits.json",
        repo_id=hub_name,
        repo_type="dataset",
        token=hf_token,
    )

    if os.path.exists(split_data_path):
        os.remove(split_data_path)

    print(f"Subido a https://huggingface.co/datasets/{hub_name}")
