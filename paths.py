import os
from pathlib import Path

root = Path(os.path.dirname(os.path.abspath(__file__)))


LS_url = "https://grothendieck.online"
bucket_url = "https://objectstorage.eu-madrid-3.oraclecloud.com/n/axfzuuzt6hgr/b/bucket-20251109-1118/o/"

# Las imágenes están directamente en bucket/(...)
images_url_path = bucket_url

# pero para las transcripciones hay una carpeta específica: bucket/transcripciones/(...)
transcripciones_url_path = bucket_url + "transcripciones%2F"


# Carpetas locales donde vamos a guardar nuestros datos
data_path = root / "data/"
data_input_path = data_path / "input/"
images_path = data_input_path / "images/"
transcriptions_path = data_input_path / "transcripciones/"

# carpetas de los exports
exports_path = data_input_path / "exports"
raw_export_filepath = exports_path / "raw_export.json"
simplified_filepath = exports_path / "simplified_export.json"
usernames_filepath = exports_path / "usernames.txt"

# carpetas donde se van a colocar los datos generados.
output_path = data_path / "output"
crops_path = output_path / "crops"


for path in [
    images_path,
    transcriptions_path,
    exports_path,
    output_path,
    crops_path,
    data_input_path,
]:
    os.makedirs(path, exist_ok=True)
