import torch
from transformers import TrainerCallback
import gc


class AdaptiveCleanupCallback(TrainerCallback):
    def __init__(self, danger_limit_gb=39.0):
        self.danger_limit_bytes = danger_limit_gb * 1024**3

    def on_step_end(self, args, state, control, **kwargs):
        reserved_memory = torch.cuda.memory_reserved()

        if reserved_memory >= self.danger_limit_bytes:
            print(
                f"\n[Warning] VRAM usage hit {reserved_memory / 1024**3:.2f}GB at step {state.global_step}. Flushing cache to prevent OOM..."
            )

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
