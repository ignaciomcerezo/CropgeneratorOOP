from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature
from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface

from typing import Callable, Generator, Any

import pandas as pd
from pathlib import Path

raw_features = Features(
    {
        "image": ImageFeature(),  # HF manejará la carga "perezosa" (Lazy Loading)
        "text": Value("string"),
        "page": Value("string"),
        "ann_id": Value("int32"),
        "order": Value("string"),
        "augment": Value("bool"),  # Flag para saber si aplicar transformaciones
        "resize_scale": Value("float32"),  # Factor de escala
        "avg_color": Sequence(
            Value("int32"), length=3
        ),  # RGB promedio de la página completa
        "context": Value("string"),
    }
)


# un generador (debemos pasárselo usando este sistema a Dataset.from_generator).
# que, a partir de un dataframe, nos da el generador de las muestras
def generate_generator(
    pdi: PairsDataInterface, augment=True, resize_scale=0.5
) -> Callable[[pd.DataFrame], Generator[dict[str, Any]]]:
    """
    Toma como entrada un PairsDataInterface, un booleano y un factor de escala 0...1.
    Devuelve una función cuya única entrada es
    Genera solamente las RUTAS y metadatos. NO abre las imágenes aquí para salvar RAM.
    HuggingFace abrirá la imagen automáticamente cuando sea necesario gracias a ImageFeature().
    """
    paths: PathBundle = pdi.paths

    def raw_data_generator(df: pd.DataFrame) -> Generator:
        # iteramos el dataframe (es rápido porque son solo textos)
        for index, row in df.iterrows():

            img_name = str(row.crop_file)
            text = str(row.text)
            row_page = str(row.page)
            row_ann_id = int(row.id)
            is_letter = row.is_letter
            avg_color = tuple([int(x) for x in row.background_color[1:-1].split(",")])

            order = str(row.order)

            dataset_subfolder = f"order{order}"

            image_path = Path(paths.dataset_path) / dataset_subfolder / img_name

            context = pdi.get_rows_context_by_words(row)

            yield {
                "image": image_path,  # nótese que no abrimos la imagen, solamente pasamos la ruta
                "text": text,
                "context": context,
                # de aquí en adelante son valores que realmente no pasamos al modelo, pero son
                # necesarios para el fit_transform o para poder ubicar qué archivo es
                # durante el análisis de los resultados.
                "ann_id": row_ann_id,
                "page": row_page,
                "order": order,
                "is_letter": is_letter,
                "augment": augment,  # pasamos el flag para usarlo luego
                "resize_scale": resize_scale,  # pasamos el factor para usarlo luego
                "avg_color": avg_color,  # color promedio, para luego usarlo
            }

    return raw_data_generator
