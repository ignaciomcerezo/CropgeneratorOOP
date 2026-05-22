import jiwer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def get_compute_metrics_function_from_tokenizer(tokenizer):
    def compute_metrics(eval_preds):
        logits_argmax, labels = eval_preds

        # Standard separator for most Unsloth/HuggingFace templates
        # Adjust this if your template uses "### Response:" or similar
        assistant_marker = "assistant\n"

        cleaned_preds = []
        cleaned_labels = []

        for i in range(len(labels)):
            # 1. Mask out -100 indices (which should be the prompt/padding)
            # We do this first to reduce the amount of text the tokenizer has to process
            mask = labels[i] != -100
            p_ids = logits_argmax[i][mask]
            l_ids = labels[i][mask]

            # 2. Decode the filtered IDs
            p_text = tokenizer.decode(p_ids, skip_special_tokens=True)
            l_text = tokenizer.decode(l_ids, skip_special_tokens=True)

            # 3. String-based slicing as a safety net
            # This removes any leftover prompt or image headers that escaped the mask
            if assistant_marker in p_text.lower():
                p_text = p_text.lower().split(assistant_marker)[-1].strip()
            if assistant_marker in l_text.lower():
                l_text = l_text.lower().split(assistant_marker)[-1].strip()

            # 4. Final cleaning: remove any trailing visual artifacts
            # (Often models output a trailing newline or EOS artifacts)
            p_text = p_text.strip()
            l_text = l_text.strip()

            # Only append if we have a valid ground truth to compare against
            if len(l_text) > 0:
                cleaned_preds.append(p_text)
                cleaned_labels.append(l_text)
            else:
                # If the label is empty after cleaning, use a placeholder to avoid jiwer error
                print("Detected empty string when calculating CER!")
                cleaned_preds.append(p_text)
                cleaned_labels.append(" ")

        print("New batch!")
        for i, (clean, pred) in enumerate(zip(cleaned_labels, cleaned_preds)):
            print(f"\tClean {i}: {clean}")
            print(f"\tPred. {i}: {pred}")

        # Calculate CER on the isolated responses
        cer_score = jiwer.cer(cleaned_labels, cleaned_preds)

        return {"cer": cer_score}

    return compute_metrics
