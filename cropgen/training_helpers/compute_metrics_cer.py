from cropgen.training_helpers.part_detector import extract_collator_markers
import jiwer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class DualMetricsCalculator:
    """
    Calcula las métricas de evaluación teniendo en cuenta que hay 2 datasets: uno con contexto y otro sin él.

    También calcula la media de ambos.
    """

    def __init__(self, tokenizer, assistant_marker: str | None = None):

        self.tokenizer = tokenizer

        if assistant_marker is None:
            _, self.assistant_marker = extract_collator_markers(tokenizer)
        else:
            self.assistant_marker = assistant_marker

        self.iteration_count = 1
        self.log_filepath = "predicciones_eval.log"
        self.buffer = None

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("=== LOG DE PREDICCIONES EN EVALUACIÓN ===\n\n")

    def __call__(self, eval_preds):

        logits_argmax, labels = eval_preds

        cleaned_preds = []
        cleaned_labels = []

        for i in range(len(labels)):
            mask = labels[i] != -100
            p_ids = logits_argmax[i][mask]
            l_ids = labels[i][mask]

            p_text = self.tokenizer.decode(p_ids, skip_special_tokens=True)
            l_text = self.tokenizer.decode(l_ids, skip_special_tokens=True)

            if self.assistant_marker in p_text.lower():
                p_text = p_text.lower().split(self.assistant_marker)[-1].strip()
            if self.assistant_marker in l_text.lower():
                f_parts = l_text.lower().split(self.assistant_marker)
                l_text = f_parts[-1].strip() if len(f_parts) > 1 else l_text.strip()

            p_text = p_text.strip()
            l_text = l_text.strip()

            if len(l_text) > 0:
                cleaned_preds.append(p_text)
                cleaned_labels.append(l_text)
            else:
                cleaned_preds.append(p_text)
                cleaned_labels.append(" ")

        current_cer = jiwer.cer(cleaned_labels, cleaned_preds)

        if self.buffer is None:
            self.buffer = {
                "preds": cleaned_preds,
                "labels": cleaned_labels,
                "cer_no_context": current_cer,
            }
            return {"cer": current_cer}

        else:
            cer_no_context = self.buffer["cer_no_context"]
            combined_cer = (cer_no_context + current_cer) / 2.0

            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"# Iteration {self.iteration_count}:\n\n")

                preds_no_ctx = self.buffer["preds"]
                preds_with_ctx = cleaned_preds
                labels_real = self.buffer["labels"]

                for i in range(len(labels_real)):
                    f.write(f"[Muestra {i:03d}]\n")
                    f.write(f"\t            real: {labels_real[i]}\n")

                    p_nc = preds_no_ctx[i] if i < len(preds_no_ctx) else "N/A"
                    p_wc = preds_with_ctx[i] if i < len(preds_with_ctx) else "N/A"

                    f.write(f"\tpredicción s/ctx: {p_nc}\n")
                    f.write(f"\tpredicción c/ctx:  {p_wc}\n\n")

                f.write(f"--- METRICS ---\n")
                f.write(f"CER s/ctx: {cer_no_context:.4f}\n")
                f.write(f"CER c/ctx: {current_cer:.4f}\n")
                f.write(f"CER COMB:  {combined_cer:.4f}\n")
                f.write("-" * 40 + "\n\n")

            self.buffer = None
            self.iteration_count += 1

            return {"cer": current_cer, "combined_cer": combined_cer}
