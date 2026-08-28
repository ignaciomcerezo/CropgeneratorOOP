from __future__ import annotations

import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from cropgen.shared.PathBundle import PathBundle


@dataclass(frozen=True)
class _PairInfo:
    key: str
    image_object: str
    transcription_object: str
    page_name: str


class OracleBucketInterface:
    """
    Interfaz de descarga con el bucket de Oracle. Utiliza rutas proporcionadas por un PathBundle.
    """

    def __init__(
        self, paths: PathBundle, bucket_url: str | None = None, online: bool = True
    ) -> None:
        if not bucket_url:
            if "BUCKET_URL" in os.environ:
                bucket_url: str = str(os.getenv("BUCKET_URL"))
            else:
                raise ValueError(
                    "O bien se pasa un bucket_url (str no vacio) o bien se tiene en las variables de entorno BUCKET_URL."
                )

        self.paths = paths
        self.bucket_url = self._normalize_bucket_url(bucket_url)
        self._timeout = 15
        self.online = online

        self.images_url_path = self.bucket_url
        self.transcripciones_url_path = self.bucket_url + urllib.parse.quote(
            "transcripciones/", safe=""
        )

    @classmethod
    def from_env(
        cls,
        paths: PathBundle,
        bucket_url: str | None = None,
        env_var: str = "BUCKET_URL",
        online: bool = True,
    ) -> "OracleBucketInterface":
        """
        Generates an instance of OracleBucketInterface taking missing data from the environment
        variables and dotenv.
        """
        try:

            load_dotenv()
        except Exception:
            print("Could not load the dotenv.")
            pass

        bucket_url = bucket_url if bucket_url is not None else os.getenv(env_var)
        if not bucket_url:
            raise ValueError(
                f"Did not find {env_var} in the .env or environment variables"
            )
        return cls(paths=paths, bucket_url=bucket_url, online=online)

    @staticmethod
    def _normalize_bucket_url(url: str) -> str:
        clean = url.strip().strip('"').strip("'")
        if not clean.endswith("/"):
            clean += "/"
        return clean

    @staticmethod
    def _normalize_key(stem: str) -> str:
        # empareja 003.png con 3.txt
        key = stem.lstrip("0")
        return key if key else "0"

    def _object_url(self, object_name: str) -> str:
        quoted_name = urllib.parse.quote(object_name, safe="")
        return self.bucket_url + quoted_name

    @staticmethod
    def _decode_transcription_bytes(raw_bytes: bytes, source_url: str) -> str:
        """decodifica siempre en utf-8"""
        try:
            return raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"No se ha podido decodificar en UTF-8 la transcripcion descargada desde {source_url}.",
            )

    def _list_bucket_objects(self) -> list[dict]:
        objects: list[dict] = []
        start: str | None = None

        while True:
            params = {"format": "json"}
            if start:
                params["start"] = start

            resp = requests.get(self.bucket_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()

            page_objects = payload.get("objects", []) or []
            objects.extend(page_objects)

            start = payload.get("nextStartWith")
            if not start:
                break

        return objects

    def _build_pairs(self, objects: list[dict]) -> list[_PairInfo]:
        images_by_key: dict[str, tuple[str, str]] = {}
        trans_by_key: dict[str, str] = {}

        for obj in objects:
            raw_name = obj.get("name")
            if not raw_name:
                continue

            decoded_name = urllib.parse.unquote(str(raw_name))
            path_str = decoded_name.replace("\\", "/")
            p = Path(path_str)
            suffix = p.suffix.lower()
            stem = p.stem

            if suffix == ".png":
                if "transcripciones/" in path_str:
                    continue
                key = self._normalize_key(stem)
                images_by_key[key] = (decoded_name, stem)
            elif suffix == ".txt" and "transcripciones/" in path_str:
                key = self._normalize_key(stem)
                trans_by_key[key] = decoded_name

        pairs: list[_PairInfo] = []
        for key in sorted(
            set(images_by_key.keys()).intersection(set(trans_by_key.keys()))
        ):
            img_obj, img_stem = images_by_key[key]
            txt_obj = trans_by_key[key]
            pairs.append(
                _PairInfo(
                    key=key,
                    image_object=img_obj,
                    transcription_object=txt_obj,
                    page_name=img_stem,
                )
            )
        return pairs

    def _needs_download(self, pair: _PairInfo) -> bool:
        local_img = self.paths.get_raw_image_path(pair.page_name)
        local_txt = self.paths.get_transcription_path(pair.page_name)

        img_ok = local_img.exists()
        txt_ok = local_txt.exists()

        return not (img_ok and txt_ok)

    def _compute_updates(self) -> list[_PairInfo]:
        objects = self._list_bucket_objects()
        pairs = self._build_pairs(objects)
        return [pair for pair in pairs if self._needs_download(pair)]

    def check_updates(self) -> list[str]:
        return [pair.page_name for pair in self._compute_updates()]

    def update(self) -> list[str]:

        if not self.online:
            return []

        pending = self._compute_updates()
        if not pending:
            return []
        print(
            f"OracleBucketInterface - Descargando imágenes y transcripciones en la carpeta {self.paths.data_in_path}"
        )

        def download_pair(pair: _PairInfo) -> str:
            with requests.Session() as session:
                txt_url = self._object_url(pair.transcription_object)
                img_url = self._object_url(pair.image_object)

                txt_resp = session.get(txt_url, timeout=self._timeout)
                txt_resp.raise_for_status()

                img_resp = session.get(img_url, timeout=self._timeout)
                img_resp.raise_for_status()

                local_txt = self.paths.get_transcription_path(pair.page_name)
                local_img = self.paths.get_raw_image_path(pair.page_name)

                transcription_text = self._decode_transcription_bytes(
                    txt_resp.content, txt_url
                )
                local_txt.write_text(transcription_text, encoding="utf-8")
                local_img.write_bytes(img_resp.content)

                Image.open(local_img).convert("L").save(local_img)

                return pair.page_name

        downloaded: list[str] = []
        with ThreadPoolExecutor() as executor:
            for name in tqdm(
                executor.map(download_pair, pending),
                total=len(pending),
                desc="OracleBucketInterface downloading...",
            ):
                downloaded.append(name)

        return downloaded
