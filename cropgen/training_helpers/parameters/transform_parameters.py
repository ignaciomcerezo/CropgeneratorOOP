from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True, slots=True, frozen=True)
class TransformParameters:
    # transform_train args

    instruction_text: str

    global_resize_scale: float  # 0.5
    max_dim: int  # 1024

    contextualize: bool
    context_probability: float  # redundant if contextualize = False
    max_context_chars: int  # 50

    min_rot: int | float  # 2
    max_rot: int | float  # 3

    shift_prop: float
    max_escala: float
    maxdist: float | int

    # TRAIN DATASET - configured_transform_train args

    augment_train: bool
    straighten_train: bool
    use_complex_rotation_interval_train: bool

    # EVAL DATASET - configured_transform_train args

    augment_eval: bool
    straighten_eval: bool
    use_complex_rotation_interval_eval: bool

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


session_transform_parameters = TransformParameters(
    instruction_text=r"Extract all text from this image in the original French. Do not translate. Format all math, equations, and variables in standard LaTeX enclosed in '$' (e.g., $G$, $\pi_1$). Output ONLY the transcribed text without any conversational filler or introductions.",
    global_resize_scale=0.5,
    max_dim=1024,
    contextualize=False,
    context_probability=0,  # redundant if contextualize = False btw
    max_context_chars=50,
    min_rot=2,
    max_rot=3,
    shift_prop=0.01,
    max_escala=0.02,
    maxdist=2,
    # TRAIN DATASET - configured_transform_train args
    augment_train=True,
    straighten_train=False,
    use_complex_rotation_interval_train=False,
    # EVAL DATASET - configured_transform_train args
    augment_eval=False,
    straighten_eval=False,
    use_complex_rotation_interval_eval=False,
)

print(session_transform_parameters.common_transform_parameters)
print(session_transform_parameters.eval_transform_parameters)
print(session_transform_parameters.train_transform_parameters)
