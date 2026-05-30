import os
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl


class EvalPredictionLoggerCallback(TrainerCallback):
    def __init__(self, eval_dataset, log_filepath="predicciones_eval.log"):
        self.eval_dataset = eval_dataset
        self.log_filepath = log_filepath

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("=== LOG DE PREDICCIONES EN EVALUACIÓN ===\n\n")

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model,
        tokenizer,
        **kwargs,
    ):
        iteration = state.global_step

        model.eval()

        with open(
            self.log_filepath,
            "append" if os.path.exists(self.log_filepath) else "w",
            encoding="utf-8",
        ) as f:
            f.write(f"# Iteration {iteration}:\n\n")

            with torch.no_grad():
                for idx, item in enumerate(self.eval_dataset):
                    page = item.get("page", f"{idx:03d}")
                    ground_truth = item.get(
                        "ground_truth", item.get("text", "N/A")
                    ).strip()

                    try:
                        pixel_values = (
                            item["pixel_values"].to("cuda").unsqueeze(0)
                            if "pixel_values" in item
                            else None
                        )
                        input_ids = (
                            item["input_ids"].to("cuda").unsqueeze(0)
                            if "input_ids" in item
                            else None
                        )

                        outputs = model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            max_new_tokens=256,
                            use_cache=True,
                        )

                        prediction = tokenizer.decode(
                            outputs[0], skip_special_tokens=True
                        ).strip()
                    except Exception as e:
                        prediction = f"[ERROR DURANTE GENERACIÓN: {str(e)}]"

                    f.write(f"[Página {page}]\n")
                    f.write(f"\t      real: {ground_truth}\n")
                    f.write(f"\tpredicción: {prediction}\n\n")

            f.write("-" * 40 + "\n\n")

        model.train()

        torch.cuda.empty_cache()
