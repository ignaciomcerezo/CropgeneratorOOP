from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True, slots=True, frozen=True)
class TransformParameters:
    # transform_train args

    instruction_text: str = (
        r"Extract all text from this image in the original French. Do not translate. Format all math, equations, and variables in standard LaTeX enclosed in '$' (e.g., $G$, $\pi_1$). Output ONLY the transcribed text without any conversational filler or introductions.",
    )

    global_resize_scale: float = 0.5
    max_dim: int = 1024

    contextualize: bool = False
    context_probability: float = 0  # redundant if contextualize = False
    max_context_chars: int = 50  # 50

    min_rot: int | float = 2  # 2
    max_rot: int | float = 3  # 3

    shift_prop: float = 0.01
    max_escala: float = 0.02
    maxdist: float | int = 2

    # TRAIN DATASET - configured_transform_train args

    augment_train: bool = True
    straighten_train: bool = False
    use_complex_rotation_interval_train: bool = False

    # EVAL DATASET - configured_transform_train args

    augment_eval: bool = False
    straighten_eval: bool = False
    use_complex_rotation_interval_eval: bool = False

    def _get_att_from_names(self, att_names: list[str]) -> dict[str, Any]:
        return {att_name: getattr(self, att_name) for att_name in att_names}

    @property
    def common_transform_parameters(self) -> dict[str, Any]:
        transform_global_parameter_names = [
            "instruction_text",
            "global_resize_scale",
            "max_dim",
            "contextualize",
            "context_probability",
            "max_context_chars",
            "min_rot",
            "max_rot",
            "shift_prop",
            "max_escala",
            "maxdist",
        ]
        return self._get_att_from_names(transform_global_parameter_names)

    @property
    def eval_transform_parameters(self) -> dict[str, Any]:
        eval_transform_parameter_names = [
            "augment_eval",
            "straighten_eval",
            "use_complex_rotation_interval_eval",
        ]
        return self._get_att_from_names(eval_transform_parameter_names)

    @property
    def train_transform_parameters(self) -> dict[str, Any]:
        train_transform_parameter_names = [
            "augment_train",
            "straighten_train",
            "use_complex_rotation_interval_train",
        ]
        return self._get_att_from_names(train_transform_parameter_names)
