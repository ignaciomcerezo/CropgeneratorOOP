import os
import torch  # ty:ignore[unresolved-import]
from transformers import (  # ty:ignore[unresolved-import]
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


class EvalPredictionLoggerCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, log_filepath="predicciones_eval.log"):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.log_filepath = log_filepath

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("=== LOG DE PREDICCIONES EN EVALUACIÓN ===\n\n")

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, model, **kwargs
    ):
        iteration = state.global_step
        model.eval()

        file_mode = "a" if os.path.exists(self.log_filepath) else "w"

        with open(self.log_filepath, file_mode, encoding="utf-8") as f:
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

                        generation_kwargs = {
                            "input_ids": input_ids,
                            "pixel_values": pixel_values,
                            "max_new_tokens": 256,
                            "use_cache": True,
                        }

                        if "image_grid_thw" in item:
                            generation_kwargs["image_grid_thw"] = (
                                item["image_grid_thw"].to("cuda").unsqueeze(0)
                            )
                        if "video_grid_thw" in item:
                            generation_kwargs["video_grid_thw"] = (
                                item["video_grid_thw"].to("cuda").unsqueeze(0)
                            )

                        outputs = model.generate(**generation_kwargs)

                        prediction = self.processor.decode(
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
