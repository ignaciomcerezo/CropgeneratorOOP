import torchvision.transforms as tvt
from PIL.Image import Image
import numpy as np
from training_helpers.parameters.transform_parameters import (
    session_transform_parameters,
)

contextualize = session_transform_parameters.contextualize
maxdist = session_transform_parameters.maxdist
global_resize_scale = session_transform_parameters.global_resize_scale
shift_prop = session_transform_parameters.shift_prop
max_dim = session_transform_parameters.max_dim
context_probability = session_transform_parameters.context_probability
max_escala = session_transform_parameters.max_escala
instruction_text = session_transform_parameters.instruction_text
min_rot = session_transform_parameters.min_rot
max_rot = session_transform_parameters.max_rot


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


def transform_test(batch):
    """Transforma una muestra del dataset en algo que pueda procesar la función de
    error CER que definimos más abajo"""
    resized_images = []

    # procesamos las imágenes del batch
    for i, (img, rotation) in enumerate(zip(batch["image"], batch["avg_rotation"])):
        # recuperamos el factor de escala guardado en el dataset

        w, h = img.size
        if w > max_dim or h > max_dim:
            scale_down = max_dim / max(w, h)
            img = img.resize((int(w * scale_down), int(h * scale_down)))

        resized_images.append(img)

    # sobreescribimos la lista de imágenes con las redimensionadas
    batch["image"] = resized_images
    return batch


def transform_train(
    batch,
    augment: bool,
    straighten: bool,
    use_complex_rotation_interval: bool,
    contextualize: bool = contextualize,
    maxdist: float = maxdist,
    global_resize_scale: float = global_resize_scale,
    shift_prop: float = shift_prop,
    max_dim: int | float = max_dim,
    context_probability: float = context_probability,
    max_escala: float = max_escala,
    instruction_text: str = instruction_text,
    min_rot: float = min_rot,
    max_rot: float = max_rot,
):
    """
    Esta función recibe un 'batch' (ej. 4 muestras) durante el entrenamiento.
    HuggingFace ya ha cargado las imágenes en batch['image'] como objetos PIL.
    Aquí aplicamos resize, augment y formateamos a chat.
    """
    formatted_messages = []
    # Iteramos sobre las muestras del batch actual
    for i in range(len(batch["image"])):
        image: Image = batch["image"][i]
        text: str = batch["text"][i]
        # color promedio
        avg_col_tuple: tuple[int, int, int] = tuple(batch["avg_color"][i])
        average_rotation: float = batch["avg_rotation"][i]
        context: str = batch["context"][i]

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
                        fill=avg_col_tuple,  # pyright: ignore[reportArgumentType]
                    ),
                    tvt.RandomAffine(
                        degrees=0,
                        translate=(shift_prop, shift_prop),
                        scale=(1 - max_escala, 1 + max_escala),
                        shear=maxdist,
                        fill=avg_col_tuple,  # pyright: ignore[reportArgumentType]
                    ),
                    tvt.RandomPerspective(
                        distortion_scale=0.05,
                        p=0.3,
                        fill=avg_col_tuple,  # pyright: ignore[reportArgumentType]
                    ),
                ]
            )

            image = current_transforms(image)

        w, h = image.size
        if w > max_dim or h > max_dim:
            scale_down = max_dim / max(w, h)
            image = image.resize((int(w * scale_down), int(h * scale_down)))

        if (
            contextualize
            and (0 < context_probability)
            and (len(context) > 0)
            and (np.random.rand() < context_probability)
        ):  # adición del contexto
            conversation = [
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
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_text},
                        {"type": "image", "image": image},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ]
        formatted_messages.append(conversation)

    # devolvemos las cosas en el formato que espera el trainer
    return {"messages": formatted_messages}


transform_train_configured = lambda batch: transform_train(
    batch,
    augment=session_transform_parameters.augment_train,
    straighten=session_transform_parameters.straighten_train,
    use_complex_rotation_interval=session_transform_parameters.use_complex_rotation_interval_train,
)

transform_eval_configured = lambda batch: transform_train(
    batch,
    augment=session_transform_parameters.augment_eval,
    straighten=session_transform_parameters.straighten_eval,
    use_complex_rotation_interval=session_transform_parameters.use_complex_rotation_interval_eval,
)
