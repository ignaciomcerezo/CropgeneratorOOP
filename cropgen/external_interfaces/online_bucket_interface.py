from __future__ import annotations
from cropgen.external_interfaces.external_interface import ExternalInterface
from typing import Literal, Callable

import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from cropgen.shared.path_bundle import PathBundle


class OnlineBucketInterface(ExternalInterface):
    """
    Bucket download interface. Uses paths provided by a PathBundle instance.
    """

    def __init__(
        self,
        paths: PathBundle,
        bucket_url: str | None = None,
        online: bool = True,
        extension_wanted: str = ".png",
        what_downloading: Literal[
            "raw_images", "background_images", "stroke_images"
        ] = "raw_images",
    ) -> None:
        if not bucket_url:
            if "BUCKET_URL" in os.environ:
                bucket_url: str = str(os.getenv("BUCKET_URL"))
            else:
                raise ValueError(
                    "Either a bucket_url is provided or one can be found in the env variables (as BUCKET_URL)."
                )

        self.paths = paths
        self.bucket_url = self._normalize_bucket_url(bucket_url)
        self._timeout = 15
        self.online = online
        self._type_downloading = what_downloading
        self.corresponding_path_accesor  # to check it is of the correct type

        if not extension_wanted.startswith("."):
            extension_wanted = f".{extension_wanted}"
        self.extension = extension_wanted

        self.images_url_path = self.bucket_url

    @property
    def corresponding_path_accesor(self) -> Callable[[str], Path]:
        match self._type_downloading:
            case "raw_images":
                return self.paths.get_raw_image_path
            case "background_images":
                return self.paths.get_background_image_path
            case "stroke_images":
                return self.paths.get_stroke_image_path
            case _:
                raise ValueError("Unsupported type_downloading")

    @classmethod
    def from_env(
        cls,
        paths: PathBundle,
        bucket_url: str | None = None,
        env_var: str = "BUCKET_URL",
        online: bool = True,
    ) -> "OnlineBucketInterface":
        """
        Generates an instance of  taking missing data from the environment
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

    def _compute_updates_from_objects(self, objects: list[dict]) -> list[str]:
        pending: dict[str, str] = {}

        for obj in objects:
            raw_name = obj.get("name")
            if not raw_name:
                continue

            decoded_name = urllib.parse.unquote(str(raw_name))
            path_str = decoded_name.replace("\\", "/")
            p = Path(path_str)
            if p.suffix.lower() != self.extension:
                continue

            page_name = p.stem
            local_img = self.corresponding_path_accesor(page_name)
            if not local_img.exists():
                pending.setdefault(page_name, decoded_name)

        return sorted(pending)

    def _compute_pending_objects(self) -> dict[str, str]:
        objects = self._list_bucket_objects()
        pending: dict[str, str] = {}

        for obj in objects:
            raw_name = obj.get("name")
            if not raw_name:
                continue

            decoded_name = urllib.parse.unquote(str(raw_name))
            path_str = decoded_name.replace("\\", "/")
            p = Path(path_str)
            if p.suffix.lower() != self.extension:
                continue

            page_name = p.stem
            local_img = self.corresponding_path_accesor(page_name)
            if not local_img.exists():
                pending.setdefault(page_name, decoded_name)

        return pending

    def _compute_updates(self) -> list[str]:
        return sorted(self._compute_pending_objects())

    def check_updates(self) -> list[str]:
        return self._compute_updates()

    def update(self) -> list[str]:
        if not self.online:
            return []

        pending = self._compute_pending_objects()
        if not pending:
            return []
        print(f" - Downloading images into {self.paths.data_in_path}")

        def download_image(item: tuple[str, str]) -> str:
            page_name, object_name = item
            with requests.Session() as session:
                img_url = self._object_url(object_name)
                img_resp = session.get(img_url, timeout=self._timeout)
                img_resp.raise_for_status()

                local_img = self.corresponding_path_accesor(page_name)
                local_img.write_bytes(img_resp.content)

                Image.open(local_img).convert("L").save(local_img)
                return page_name

        downloaded: list[str] = []
        with ThreadPoolExecutor() as executor:
            for name in tqdm(
                executor.map(download_image, pending.items()),
                total=len(pending),
                desc=" downloading...",
            ):
                downloaded.append(name)

        return downloaded

    def parts_managed(
        self,
    ) -> tuple[Literal["raw_images", "background_images", "stroke_images"],]:
        return (self._type_downloading,)

    def parts_required(self):
        return []

    def setup(self) -> None:
        """Download pending bucket images if the interface is online."""
        if not self.online:
            return
        self.update()
        return
