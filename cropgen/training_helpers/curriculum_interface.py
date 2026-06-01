import torch  # ty:ignore[unresolved-import]
from cropgen.training_helpers.restrict_length import restrict_length
from transformers import TrainerCallback  # ty:ignore[unresolved-import]
from cropgen.training_helpers.parameters.curriculum_params import CurriculumParams
import os

ln = "\n"
tab = "\t"


class CurriculumDatasetInterface(torch.utils.data.Dataset):
    def __init__(
        self, full_dataset, CurriculumParams, restrict_length_fn, transform_func
    ):
        self.full_dataset = full_dataset
        self.orders_per_epoch = CurriculumParams.orders_per_epoch
        self.restrict_length_fn = restrict_length_fn
        self.transform_func = transform_func
        self.current_stage_idx = 0
        self.active_dataset = None

        self.update_for_stage(0)

    def update_for_stage(self, stage_idx):
        if stage_idx < len(self.orders_per_epoch):
            self.current_stage_idx = stage_idx
            acceptable_lengths = self.orders_per_epoch[stage_idx]

            self.active_dataset = self.restrict_length_fn(
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
        return self.active_dataset[idx]

    def __getattr__(self, name):
        return getattr(self.active_dataset, name)


def calculate_total_curriculum_steps(
    full_dataset, schedule, per_device_batch_size, grad_accum
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    total_batch_size = per_device_batch_size * grad_accum * world_size
    total_steps = 0

    original_transform = full_dataset.format["transform"]
    full_dataset.set_transform(None)

    raw_orders = full_dataset["order"]
    orders_str = [str(o) for o in raw_orders]

    full_dataset.set_transform(original_transform)

    for lengths in schedule:
        acceptable_set = set(str(x) for x in lengths)

        num_samples = sum(1 for o in orders_str if o in acceptable_set)

        num_batches = num_samples // total_batch_size
        total_steps += num_batches

    return total_steps


class CurriculumCallback(TrainerCallback):
    def __init__(self, dataset_proxy):
        self.dataset_proxy = dataset_proxy

    def on_epoch_end(self, args, state, control, **kwargs):
        next_stage = self.dataset_proxy.current_stage_idx + 1
        if next_stage < len(self.dataset_proxy.schedule):
            self.dataset_proxy.update_for_stage(next_stage)
        else:
            print(f"{ln}[Currículum] Etapa final.")
