from __future__ import annotations
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
import os
import urllib.parse

import requests

from cropgen.shared.PathBundle import PathBundle


@dataclass(frozen=True)
class _PairInfo:
    key: str
    image_object: str
    transcription_object: str
    image_stem: str
    image_size: int | None
    transcription_size: int | None


class OracleBucketInterface:
    def __init__(
        self,
        paths: PathBundle,
        bucket_url: str,
    ) -> None:
        if not bucket_url or not isinstance(bucket_url, str):
            raise ValueError("bucket_url debe ser un str no vacio")

        self.paths = paths
        self.bucket_url = self._normalize_bucket_url(bucket_url)
        self._timeout = 15

        self.images_url_path = self.bucket_url
        self.transcripciones_url_path = self.bucket_url + urllib.parse.quote(
            "transcripciones/", safe=""
        )

    @classmethod
    def from_env(
        cls,
        paths: PathBundle,
        env_var: str = "BUCKET_URL",
    ) -> "OracleBucketInterface":
        # cargamos nuestro .env si python-dotenv esta disponible; si no, usa os.getenv
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass

        bucket_url = os.getenv(env_var)
        if not bucket_url:
            raise ValueError(f"No se encontro {env_var} en variables de entorno/.env")
        return cls(paths=paths, bucket_url=bucket_url)

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
        images_by_key: dict[str, tuple[str, str, int | None]] = {}
        trans_by_key: dict[str, tuple[str, int | None]] = {}

        for obj in objects:
            raw_name = obj.get("name")
            if not raw_name:
                continue

            # como puede venir con %2F en algunos casos, lo normalizamos.
            decoded_name = urllib.parse.unquote(str(raw_name))
            path_str = decoded_name.replace("\\", "/")
            p = Path(path_str)
            suffix = p.suffix.lower()
            stem = p.stem
            size = obj.get("size")
            size = (
                int(size)
                if isinstance(size, int) or (isinstance(size, str) and size.isdigit())
                else None
            )

            if suffix == ".png":
                # ignoramos fotos dentro de transcripciones
                if "transcripciones/" in path_str:
                    continue
                key = self._normalize_key(stem)
                # si hay duplicados de clave, nos quedamos con uno determinista.
                current = images_by_key.get(key)
                candidate = (decoded_name, stem, size)
                if current is None or candidate[0] < current[0]:
                    images_by_key[key] = candidate

            elif suffix == ".txt" and "transcripciones/" in path_str:
                key = self._normalize_key(stem)
                current_t = trans_by_key.get(key)
                candidate_t = (decoded_name, size)
                if current_t is None or candidate_t[0] < current_t[0]:
                    trans_by_key[key] = candidate_t

        pairs: list[_PairInfo] = []
        for key in sorted(images_by_key.keys() & trans_by_key.keys()):
            img_obj, img_stem, img_size = images_by_key[key]
            txt_obj, txt_size = trans_by_key[key]
            pairs.append(
                _PairInfo(
                    key=key,
                    image_object=img_obj,
                    transcription_object=txt_obj,
                    image_stem=img_stem,
                    image_size=img_size,
                    transcription_size=txt_size,
                )
            )
        return pairs

    def _needs_download(self, pair: _PairInfo) -> bool:
        local_img = self.paths.images_path / f"{pair.image_stem}.png"
        local_txt = self.paths.transcriptions_path / f"{pair.image_stem}.txt"

        img_ok = local_img.exists()
        txt_ok = local_txt.exists()

        return not (img_ok and txt_ok)

    def _compute_updates(self) -> list[_PairInfo]:
        objects = self._list_bucket_objects()
        pairs = self._build_pairs(objects)
        return [pair for pair in pairs if self._needs_download(pair)]

    def check_updates(self) -> list[str]:
        return [pair.image_stem for pair in self._compute_updates()]

    def update(self) -> list[str]:
        pending = self._compute_updates()
        if not pending:
            return []

        downloaded: list[str] = []
        with requests.Session() as session:
            for pair in tqdm(pending, desc="OracleBucketInterface downloading..."):
                txt_url = self._object_url(pair.transcription_object)
                img_url = self._object_url(pair.image_object)

                txt_resp = session.get(txt_url, timeout=self._timeout)
                txt_resp.raise_for_status()

                img_resp = session.get(img_url, timeout=self._timeout)
                img_resp.raise_for_status()

                local_txt = self.paths.transcriptions_path / f"{pair.image_stem}.txt"
                local_img = self.paths.images_path / f"{pair.image_stem}.png"

                local_txt.write_text(txt_resp.text, encoding="utf-8")
                local_img.write_bytes(img_resp.content)

                downloaded.append(pair.image_stem)

        return downloaded
