from dataclasses import dataclass
from pathlib import Path

tab = "\t"
newline = "\n"


@dataclass(kw_only=True, slots=True)
class _TruthAndPredictionPair:
    ground_truth: str
    prediction: str


class EvalPredictionLogger:
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.results: dict[int, list[_TruthAndPredictionPair]] = dict()
        self._iteration_number: int = -1

    @property
    def current_iteration(self):
        return self._iteration_number

    @current_iteration.setter
    def current_iteration(self, iteration_number):
        if iteration_number in self.results:
            raise ValueError(f"Ya se han guardado resultados para {iteration_number}.")
        self.results[iteration_number] = []
        self._iteration_number = iteration_number

    def add_entry(self, ground_truth: str, prediction: str):
        self.results[self.current_iteration].append(
            _TruthAndPredictionPair(ground_truth=ground_truth, prediction=prediction)
        )

    def savefile(self):
        output_str = ""
        for round_number in sorted(list(self.results.keys())):
            output_str += f"{newline} # Iteración {round_number}{newline}"
            for result in self.results[round_number]:
                output_str += f"{tab} Ground truth: {result.ground_truth}{newline}"
                output_str += f"{tab}   Prediction: {result.prediction}{newline}"

        self.filepath.write_text(output_str)
