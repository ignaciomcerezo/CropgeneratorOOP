import re
import requests
import os
from parameters import images_url_path, transcripciones_url_path
from kaggle_integration.PathBundle import PathBundle
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm


def download_single_img_txt_pair(paths: PathBundle, image_string, session):
    """
    Función para descargar una sola imagen y su transcripción - la usaremos en paralelo.
    El argumento "session" se emplea para reducir la latencia. En lugar de abrir y cerrar
    una para cada archivo, se emplea la misma para todos.
    """
    try:
        # preparamos las URLs
        image_url = images_url_path + image_string + ".png"
        transcription_name = re.sub(
            r"^0+", "", image_string
        )  # las transcripciones no tienen el relleno de ceros a la izquierda

        transcription_url = transcripciones_url_path + transcription_name + ".txt"

        image_save_path = os.path.join(paths.images_path, image_string + ".png")
        transcription_save_path = os.path.join(
            paths.transcriptions_path, image_string + ".txt"
        )

        # primero buscamos las transcripciones y luego las imágenes: puesto que
        # hay más imágenes que transcripciones (hay páginas que no están transcritas)
        # lo mejor es pedir la transcripción y, si falla, pasar a la siguiente,
        # pues la imagen no serviría de nada. Al revés no funciona, puesto que hay
        # imágenes para t0do.

        # cogemos las transcripciones (falla si no hay transcripción)
        trans_resp = session.get(transcription_url, timeout=10)
        trans_resp.raise_for_status()
        transcription_text = trans_resp.text

        # cargamos la imagen (solamente si la transcripción está, así ahorramos tiempo)
        img_resp = session.get(image_url, timeout=10)
        img_resp.raise_for_status()
        image_content = img_resp.content

        # guardamos los archivos
        with open(transcription_save_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)

        with open(image_save_path, "wb") as f:
            f.write(image_content)

        return (0, None)  # no hay errores

    except requests.exceptions.HTTPError:
        return (1, image_string)  # error 1: no hay transcripción
    except Exception as e:
        return (2, image_string)  # error 2: no se ha podido procesar la imagen


def download_all(paths: PathBundle, force_download: bool = False):
    # buscamos los objetos que hay en nuestro bucket (las fotos únicamente)
    data_response = requests.get(images_url_path, params={"format": "json"}).json()
    bucket_file_names = [obj["name"] for obj in data_response["objects"]]

    image_strings = sorted(
        a.rsplit(".", 1)[0]
        for a in bucket_file_names
        if ((len(a.rsplit(".", 1)) > 1) and (a.rsplit(".", 1)[1] == "png"))
    )

    print(f"Se han encontrado {len(image_strings)} imágenes. Descargando...")

    # usamos t0do el rato la misma petición al servidor, así
    # nos quitamos el overhead de tener que abrir y cerrar
    # conexiones para cada archivo, lo que reduce la latencia, ya que estamos
    # descargando todos los archivos del mismo sitio (nuestro bucket de oracle)

    errors = [
        [],
        [],
    ]  # manejamos errores de tipo 1 -índice 0- (sin transcripción); el resto van al tipo 2 -índice 1-

    # descargamos solamente las imágenes que no tengamos (o todas, si nos falta alguna).
    images_to_download = [
        img_str
        for img_str in image_strings
        if (force_download or not (paths.images_path / f"{img_str}.png").exists())
    ]
    with requests.Session() as session:
        # max_workers es el número máximo de archivos que descargamos a la vez
        with ThreadPoolExecutor(max_workers=10) as executor:
            # añadimos todas las tareas (todas las descargas) al pool,
            # con la misma session
            futures = [
                executor.submit(download_single_img_txt_pair, paths, img_str, session)
                for img_str in images_to_download
            ]

            # "recogemos" los resultados de los diferentes workers, pero en el orden en
            # el que los hemos llamado.
            for future in tqdm(futures):
                err, img_name = future.result()
                if err:
                    errors[err - 1].append(img_name)
    if len(errors[0]) > 0:
        print(
            f"Las siguientes imágenes no tienen transcripción asociada: {" ".join(errors[0])}"
        )
    if len(errors[1]) > 0:
        print(
            f"Las siguientes imágenes han tenido otro tipo de error durante la descarga: {" ".join(errors[1])}"
        )
