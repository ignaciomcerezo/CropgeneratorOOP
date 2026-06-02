from typing import Callable
import torch  # ty:ignore[unresolved-import]
from transformers import TrainerCallback  # ty:ignore[unresolved-import]
from cropgen.training_helpers.parameters.curriculum_params import CurriculumParams
import os
from datasets import Dataset

ln = "\n"
tab = "\t"


class CurriculumDatasetInterface(torch.utils.data.Dataset):
    def __init__(
        self, full_dataset, CurriculumParams, restrict_length_fn, transform_func
    ):
        self.full_dataset: Dataset = full_dataset
        self.orders_per_epoch: list[list[int | str]] = CurriculumParams.orders_per_epoch
        self.restrict_length_fn: Callable = restrict_length_fn
        self.transform_func: Callable = transform_func
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


def calculate_total_curriculum_steps(
    full_dataset, schedule, per_device_batch_size, grad_accum
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    total_batch_size = per_device_batch_size * grad_accum * world_size
    total_steps = 0

    raw_orders: list[str | int] = full_dataset.data["order"].to_pylist()
    orders_str: list[str] = [str(o) for o in raw_orders]

    # Calculate steps across your curriculum schedule
    for lengths in schedule:
        acceptable_set = set(str(x) for x in lengths)
        num_samples = sum(1 for o in orders_str if o in acceptable_set)

        num_batches: int = num_samples // total_batch_size
        total_steps += num_batches

    return total_steps


class CurriculumCallback(TrainerCallback):
    def __init__(self, curriculum_dataset_interface: CurriculumDatasetInterface):
        self.curriculum_dataset_interface = curriculum_dataset_interface

    def on_epoch_end(self, args, state, control, **kwargs):
        next_stage: int = self.curriculum_dataset_interface.current_stage_idx + 1
        if next_stage < len(self.curriculum_dataset_interface.orders_per_epoch):
            self.curriculum_dataset_interface.update_for_stage(next_stage)
        else:
            print(f"{ln}[Currículum] Etapa final.")
