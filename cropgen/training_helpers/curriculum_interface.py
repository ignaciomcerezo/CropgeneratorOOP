from typing import Callable
import torch  # ty:ignore[unresolved-import]
from transformers import TrainerCallback  # ty:ignore[unresolved-import]
from cropgen.training_helpers.parameters.curriculum_params import CurriculumParams
import os
from datasets import Dataset

ln = "\n"
tab = "\t"


class CurriculumDatasetInterface(torch.utils.data.Dataset):
    """
    Interfaz con el dataset que gestiona automáticamente el entrenamiento por currículum. Requiere:
        - Un dataset
        - Una instancia de CurriculumParam
        - Una función de filtrado para restringir el dataset completo.
        - La transformación que se aplica a cada subdataset.
    """

    def __init__(
        self,
        full_dataset: Dataset,
        CurriculumParams: CurriculumParams,
        restrict_length_fn: Callable,
        transform_func: Callable,
    ):
        self.full_dataset = full_dataset
        self.orders_per_epoch = CurriculumParams.orders_per_epoch
        self.restrict_length_fn = restrict_length_fn
        self.transform_func = transform_func
        self.current_stage_idx: int = 0
        self.active_dataset: Dataset | None = None

        self.update_for_stage(0)

    def update_for_stage(self, stage_idx: int):
        if stage_idx < len(self.orders_per_epoch):
            self.current_stage_idx = stage_idx
            acceptable_lengths: list[int | str] = self.orders_per_epoch[stage_idx]

            self.active_dataset: Dataset = self.restrict_length_fn(
                self.full_dataset,
                acceptable_lengths=acceptable_lengths,
                transform_func=self.transform_func,
            )
            print(
                f"[Curriculum] Paso {stage_idx}:{ln}{tab}Tamaño del dataset: {len(self.active_dataset)} muestras{ln}{tab}Ordenes: {acceptable_lengths}"
            )

    def __len__(self):
        return len(self.active_dataset)

    def __getitem__(self, idx):
        if self.active_dataset is None:
            raise ValueError("No hay dataset activo")
        else:
            return self.active_dataset[idx]

    def __getattr__(self, name):

        return getattr(self.active_dataset, name)

    def update_for_orders(self, acceptable_lengths: list[int | str]):
        self.active_dataset = self.restrict_length_fn(
            self.full_dataset,
            acceptable_lengths=acceptable_lengths,
            transform_func=self.transform_func,
        )
        print(
            f"[Curriculum] Phase parameters updated.\n\tDataset size: {len(self.active_dataset)} samples\n\tOrders: {acceptable_lengths}"
        )

    def calculate_total_curriculum_steps(
        self, per_device_batch_size: int, grad_accum: int
    ):
        """
        Calcula el número total de iteraciones necesarias para realizar el entrenamiento por currículum.
        """
        full_dataset: Dataset = self.full_dataset
        orders_per_epoch = self.orders_per_epoch

        total_batch_size = per_device_batch_size * grad_accum
        total_steps = 0

        raw_orders: list[str | int] = full_dataset.data["order"].to_pylist()
        orders_str: list[str] = [str(o) for o in raw_orders]

        for lengths in orders_per_epoch:
            acceptable_set = set(str(x) for x in lengths)
            num_samples = sum(1 for o in orders_str if o in acceptable_set)

            num_batches: int = num_samples // total_batch_size
            total_steps += num_batches

        return total_steps
