from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(kw_only=True, slots=True, frozen=True)
class AdapterParameters:
    model: str
    finetune_vision_layers: bool = False  # False if not finetuning vision layers
    finetune_language_layers: bool = True  # False if not finetuning language layers
    finetune_attention_modules: bool = True  # False if not finetuning attention layers
    finetune_mlp_modules: bool = True  # False if not finetuning MLP layers
    r: int = 8  # The larger, the higher the accuracy, but might overfit
    lora_alpha: int = 8  # Recommended alpha == r at least
    lora_dropout: float = 0.05
    bias: Any = "none"
    random_state: int = 3407
    use_rslora: bool = False  # We support rank stabilized LoRA
    loftq_config: Any = None  # And LoftQ
