from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class CurriculumParams:
    orders_per_epoch: list[list[int | str]]
