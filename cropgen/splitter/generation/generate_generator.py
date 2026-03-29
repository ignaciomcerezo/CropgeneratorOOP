from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature
from cropgen.shared.PathBundle import PathBundle
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pandas as pd

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

    page2fulltext = dict()

    pages = pd.unique(pdi.pages)

    full = pdi.df[pdi.df.page == "full"]

    for page in pages:
        page2fulltext[page] = full[full.page == page].text

    def raw_data_generator():
        # iteramos el dataframe (es rápido porque son solo textos)
        for index, row in pdi.df.iterrows():

            img_name = row.crop_file
            text = row.text
            page = row.page
            is_letter = row.is_letter
            s_index = row.sindex
            avg_color = tuple([int(x) for x in row.background_color[1:-1].split(",")])

            order = str(row.order)

            dataset_subfolder = f"order{order}"

            image_path = Path(paths.dataset_path) / dataset_subfolder / img_name

            full_text = page2fulltext[page]

            if row.has_enough_context:
                if s_index >= max_context_chars:
                    # si hay suficiente contexto en la página, usamos solamente esa
                    context = full_text[s_index - max_context_chars : s_index].strip()
                else:
                    # si no, hay que tirar de la anterior (sabemos que esto no da problemas en general)
                    prev_full_text = page2fulltext[prev_page(page)]
                    context = (
                        prev_full_text[-(max_context_chars - s_index) :].strip()
                        + " "
                        + full_text[:s_index].strip()
                    )

            else:
                context = ""  # no hay contexto posible!

            yield {
                "image": image_path,  # nótese que no abrimos la imagen, solamente pasamos la ruta
                "text": text,
                "context": context,
                # de aquí en adelante son valor que realmente no pasamos al modelo, pero son
                # necesarios para el fit_transform o para poder ubicar qué archivo es
                # durante el análisis de los resultados.
                "page": page,
                "order": order,
                "is_letter": is_letter,
                "augment": augment,  # pasamos el flag para usarlo luego
                "resize_scale": resize_scale,  # pasamos el factor para usarlo luego
                "avg_color": avg_color,  # color promedio, para luego usarlo
            }

    return raw_data_generator
