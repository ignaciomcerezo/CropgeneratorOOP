import torchvision.transforms as tvt  # ty:ignore[unresolved-import]
from PIL.Image import Image
import numpy as np
from typing import Callable, Literal


def _choose_rotation_interval_simple(added_rotation) -> tuple[float, float]:
    return -added_rotation, added_rotation


def _choose_rotation_interval_complex(
    average_rotation: float,
    min_added_rotation: float,
    max_added_rotation: float,
) -> tuple[float, float]:
    assert min_added_rotation > 0
    assert max_added_rotation > 0

    magnitude = abs(average_rotation)

    if magnitude < min_added_rotation:
        return -min_added_rotation, min_added_rotation

    elif average_rotation < max_added_rotation / 2:
        if average_rotation < 0:
            return 0, 2 * magnitude
        else:
            return -2 * magnitude, 0
    else:
        if average_rotation < 0:
            return 0, 2 * max_added_rotation
        else:
            return -2 * max_added_rotation, 0


def transform_test(batch, max_dim: int = 1024):
    """Transforma una muestra del dataset en algo que pueda procesar la función de
    error CER que definimos más abajo"""
    resized_images = []

    # procesamos las imágenes del batch
    for i, (img, rotation) in enumerate(zip(batch["image"], batch["avg_rotation"])):
        # recuperamos el factor de escala guardado en el dataset

        w, h = img.size
        if w > max_dim or h > max_dim:
            scale_down: float = max_dim / max(w, h)
            img: Image = img.resize((int(w * scale_down), int(h * scale_down)))

        resized_images.append(img)

    # sobreescribimos la lista de imágenes con las redimensionadas
    batch["image"] = resized_images
    return batch


def transform_train(
    batch,
    augment: bool,
    straighten: bool,
    use_complex_rotation_interval: bool,
    maxdist: float,
    global_resize_scale: float,
    shift_prop: float,
    max_dim: int | float,
    context_probability: float,
    max_escala: float,
    instruction_text: str,
    min_rot: float,
    max_rot: float,
    context_mode: Literal["never", "always", "probabilistic"],
    min_context: int,
    max_context: int,
) -> dict[str, list]:
    """
    Recibe un batch de muestras durante el entrenamiento o evaluación.
    Aplica transformaciones de imagen y le da formato a los datos
    en base al modo de contexto configurado.
    """
    formatted_messages = []

    for i in range(len(batch["image"])):
        image: Image = batch["image"][i]
        text: str = batch["text"][i]
        avg_col_tuple: tuple[int, int, int] = tuple(batch["avg_color"][i])
        average_rotation: float = batch["avg_rotation"][i]
        context: str = batch.get("context", [""] * len(batch["image"]))[i]

        image = image.convert("RGB")

        if global_resize_scale != 1:
            w, h = image.size
            image = image.resize(
                (int(w * global_resize_scale), int(h * global_resize_scale))
            )

        if straighten:
            image = image.rotate(
                -average_rotation, expand=True, fillcolor=avg_col_tuple
            )
            average_rotation = 0

        if augment:
            if use_complex_rotation_interval:
                rotation_interval = _choose_rotation_interval_complex(
                    average_rotation=average_rotation,
                    min_added_rotation=min_rot,
                    max_added_rotation=max_rot,
                )
            else:
                rotation_interval = _choose_rotation_interval_simple(
                    added_rotation=min_rot
                )

            current_transforms = tvt.Compose(
                [
                    tvt.RandomRotation(
                        degrees=rotation_interval,
                        expand=False,
                        fill=avg_col_tuple,
                    ),
                    tvt.RandomAffine(
                        degrees=0,
                        translate=(shift_prop, shift_prop),
                        scale=(1 - max_escala, 1 + max_escala),
                        shear=maxdist,
                        fill=avg_col_tuple,
                    ),
                    tvt.RandomPerspective(
                        distortion_scale=0.05,
                        p=0.3,
                        fill=avg_col_tuple,
                    ),
                ]
            )
            image = current_transforms(image)

        w, h = image.size
        if w > max_dim or h > max_dim:
            scale_down = max_dim / max(w, h)
            image = image.resize((int(w * scale_down), int(h * scale_down)))

        is_context_valid = (context is not None) and (min_context <= len(context))

        context_length = np.random.randint(min_context, max_context)

        # aleatorizamos la cantidad de contexto añadida también
        context = context[:context_length]

        no_context_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

        with_context_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"For reference, here is the previous text: {context}",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

        if context_mode == "never":
            formatted_messages.append(no_context_conv)

        elif context_mode == "always":
            formatted_messages.append(
                with_context_conv if is_context_valid else no_context_conv
            )

        elif context_mode == "probabilistic":
            if is_context_valid and np.random.rand() < context_probability:
                formatted_messages.append(with_context_conv)
            else:
                formatted_messages.append(no_context_conv)

    return {"messages": formatted_messages}
