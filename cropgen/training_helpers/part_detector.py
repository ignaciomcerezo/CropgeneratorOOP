def extract_collator_markers(tokenizer, verbose=False) -> tuple[str, str]:
    """
    Extrae los marcadores de asistente y usuario del tokenizador de forma automática.
    """
    # Buscando la parte de instrucción y de respuesta de un tokenizer y modelo automáticamente
    u1, a1, u2 = "USUARIO_1_EJEMPLO", "MODELO_1_EJEMPLO", "USUARIO_2_EJEMPLO"

    # Isolate response_part via generation prompt delta
    t_false = tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}], tokenize=False, add_generation_prompt=False
    )
    t_true = tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}], tokenize=False, add_generation_prompt=True
    )
    response_part = t_true[len(t_false) :]

    t_two = tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}],
        tokenize=False,
        add_generation_prompt=False,
    )
    asst_closer = t_two[t_two.rfind(a1) + len(a1) :]

    t_three = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    between = t_three[t_three.find(a1) + len(a1) : t_three.find(u2)]

    instruction_part = (
        between[len(asst_closer) :] if between.startswith(asst_closer) else between
    )

    if verbose:
        ln = "\n"
        backslash_n = "\\n"
        print(f"Instruction part: {instruction_part.replace(ln, backslash_n)}")
        print(f"Response part: {response_part.replace(ln, backslash_n)}")

    return instruction_part, response_part
