from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature
from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pandas as pd
from pathlib import Path

raw_features = Features(
    {
        "image": ImageFeature(),  # HF manejará la carga "perezosa" (Lazy Loading)
        "text": Value("string"),
        "page": Value("string"),
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
    paths: PathBundle, pdi: PairsDataInterface, augment=True, resize_scale=0.5
):
    """
    Genera solamente las RUTAS y metadatos. NO abre las imágenes aquí para salvar RAM.
    HuggingFace abrirá la imagen automáticamente cuando sea necesario gracias a ImageFeature().
    """

    def raw_data_generator():
        # iteramos el dataframe (es rápido porque son solo textos)
        for index, row in pdi.df.iterrows():

            img_name = row.crop_file
            text = row.text
            row_page = row.page
            row_ann_id = row.id
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
                # de aquí en adelante son valor que realmente no pasamos al modelo, pero son
                # necesarios para el fit_transform o para poder ubicar qué archivo es
                # durante el análisis de los resultados.
                "page": row_page,
                "order": order,
                "is_letter": is_letter,
                "augment": augment,  # pasamos el flag para usarlo luego
                "resize_scale": resize_scale,  # pasamos el factor para usarlo luego
                "avg_color": avg_color,  # color promedio, para luego usarlo
            }

    return raw_data_generator
