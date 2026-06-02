from cropgen.training_helpers.part_detector import extract_collator_markers
import jiwer
import os


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def get_compute_metrics_function_from_tokenizer(
    tokenizer, assistant_marker: str | None = None
):

    if assistant_marker is None:
        _, assistant_marker = extract_collator_markers(tokenizer)

    def compute_metrics(eval_preds):
        logits_argmax, labels = eval_preds
        log_filepath = "predicciones_eval.log"
        compute_metrics.iteration_count: int

        if not hasattr(compute_metrics, "iteration_count"):
            compute_metrics.iteration_count: int = 1
            with open(log_filepath, "w", encoding="utf-8") as f:
                f.write("=== LOG DE PREDICCIONES EN EVALUACIÓN ===\n\n")
        else:
            compute_metrics.iteration_count += 1

        cleaned_preds = []
        cleaned_labels = []

        for i in range(len(labels)):
            mask = labels[i] != -100
            p_ids = logits_argmax[i][mask]
            l_ids = labels[i][mask]

            p_text = tokenizer.decode(p_ids, skip_special_tokens=True)
            l_text = tokenizer.decode(l_ids, skip_special_tokens=True)

            if assistant_marker in p_text.lower():
                p_text = p_text.lower().split(assistant_marker)[-1].strip()
            if assistant_marker in l_text.lower():
                f_parts = l_text.lower().split(assistant_marker)
                l_text = f_parts[-1].strip() if len(f_parts) > 1 else l_text.strip()

            p_text = p_text.strip()
            l_text = l_text.strip()

            if len(l_text) > 0:
                cleaned_preds.append(p_text)
                cleaned_labels.append(l_text)
            else:
                cleaned_preds.append(p_text)
                cleaned_labels.append(" ")

        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(f"# Iteration {compute_metrics.iteration_count}:\n\n")
            for idx, (clean, pred) in enumerate(zip(cleaned_labels, cleaned_preds)):
                f.write(f"[Página {idx:03d}]\n")
                f.write(f"\t      real: {clean}\n")
                f.write(f"\tpredicción: {pred}\n\n")
            f.write("-" * 40 + "\n\n")

        cer_score = jiwer.cer(cleaned_labels, cleaned_preds)
        return {"cer": cer_score}

    return compute_metrics
