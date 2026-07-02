from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class CurriculumParams:
    """Parámetros para el aprendizaje por currículum"""

    orders_per_epoch: list[list[int | str]]

    def phases(self):
        return _convert_schedule_to_phases(self.orders_per_epoch)


def _convert_schedule_to_phases(schedule: list[list[int | str]]) -> list[dict]:
    if not schedule:
        return []

    phases = []
    current_orders = schedule[0]
    epochs_count = 1

    for orders in schedule[1:]:
        if orders == current_orders:
            epochs_count += 1
        else:
            phases.append({"epochs": epochs_count, "orders": current_orders})
            current_orders = orders
            epochs_count = 1

    phases.append({"epochs": epochs_count, "orders": current_orders})
    return phases
