import json
import math
from pathlib import Path
from typing import Iterable

from pysm_lib.pysm_progress_reporter import tqdm


class Siglip2OnnxDownloader:
    ONNX_BASE_URL = (
        "https://huggingface.co/onnx-community/"
        "siglip2-so400m-patch14-384-ONNX/resolve/main/onnx"
    )
    TOKENIZER_BASE_URL = (
        "https://huggingface.co/google/"
        "siglip2-so400m-patch14-384/resolve/main"
    )
    TOKENIZER_FILES = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )

    def ensure_vision_model(self, model_dir: Path, vision_model: str):
        self._ensure_files(
            target_dir=model_dir,
            base_url=self.ONNX_BASE_URL,
            filenames=(vision_model,),
            manifest_name="download_manifest_vision.json",
        )

    def ensure_text_model(self, model_dir: Path, text_model: str):
        self._ensure_files(
            target_dir=model_dir,
            base_url=self.ONNX_BASE_URL,
            filenames=(text_model, f"{text_model}_data"),
            manifest_name="download_manifest_text.json",
        )

    def ensure_tokenizer(self, tokenizer_dir: Path):
        self._ensure_files(
            target_dir=tokenizer_dir,
            base_url=self.TOKENIZER_BASE_URL,
            filenames=self.TOKENIZER_FILES,
            manifest_name="download_manifest_tokenizer.json",
        )

    def _ensure_files(
        self,
        target_dir: Path,
        base_url: str,
        filenames: Iterable[str],
        manifest_name: str,
    ):
        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        manifest_path = target_dir / manifest_name
        manifest = self._load_manifest(manifest_path)

        for filename in filenames:
            path = target_dir / filename
            url = f"{base_url}/{filename}"
            expected_size = self._manifest_file_size(manifest, filename)
            if path.exists() and self._is_existing_file_valid(path, expected_size):
                continue
            if path.exists() and expected_size <= 0:
                expected_size = self._get_remote_file_size_lazy(url)
                if self._is_existing_file_valid(path, expected_size):
                    continue
            self._download_file(
                url=url,
                target_path=path,
                expected_size=expected_size,
            )
            downloaded.append(filename)

        if downloaded:
            self._write_manifest(target_dir / manifest_name, base_url, filenames)

    def _download_file(self, url: str, target_path: Path, expected_size: int = 0):
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("requests is required to download SigLIP2 ONNX files") from e

        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        total = expected_size or self._get_remote_file_size(requests, url)

        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            total = total or self._get_response_file_size(response)
            desc = f"Downloading {target_path.name}"
            progress_unit_size = 1024 * 1024 if total > 1024 * 1024 else 1
            progress_total = (
                math.ceil(total / progress_unit_size)
                if total > 0
                else 0
            )
            progress_unit = "MiB" if progress_unit_size > 1 else "B"
            downloaded = 0
            reported_units = 0

            with tmp_path.open("wb") as f:
                with tqdm(total=progress_total, desc=desc, unit=progress_unit) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)

                        if progress_total <= 0:
                            pbar.update(len(chunk))
                            continue

                        downloaded += len(chunk)
                        current_units = min(
                            downloaded // progress_unit_size,
                            progress_total,
                        )
                        if current_units > reported_units:
                            pbar.update(current_units - reported_units)
                            reported_units = current_units

                    if progress_total > reported_units:
                        pbar.update(progress_total - reported_units)

        if total > 0 and tmp_path.stat().st_size != total:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file size mismatch for {target_path.name}: "
                f"{tmp_path.stat().st_size if tmp_path.exists() else 0} != {total}"
            )

        tmp_path.replace(target_path)

    def _load_manifest(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _manifest_file_size(self, manifest: dict, filename: str) -> int:
        try:
            return int(manifest.get("files", {}).get(filename, {}).get("size") or 0)
        except (TypeError, ValueError):
            return 0

    def _is_existing_file_valid(self, path: Path, expected_size: int) -> bool:
        if not path.exists():
            return False
        size = path.stat().st_size
        if expected_size > 0:
            return size == expected_size
        return size > 0

    def _get_remote_file_size_lazy(self, url: str) -> int:
        try:
            import requests
        except ImportError:
            return 0
        return self._get_remote_file_size(requests, url)

    def _get_remote_file_size(self, requests_module, url: str) -> int:
        try:
            response = requests_module.head(
                url,
                allow_redirects=True,
                timeout=(15, 60),
            )
            response.raise_for_status()
        except Exception:
            return 0
        return self._get_response_file_size(response)

    def _get_response_file_size(self, response) -> int:
        responses = [*getattr(response, "history", []), response]
        for item in reversed(responses):
            for header_name in ("content-length", "x-linked-size"):
                value = item.headers.get(header_name)
                if not value:
                    continue
                try:
                    size = int(value)
                except ValueError:
                    continue
                if size > 0:
                    return size
        return 0

    def _write_manifest(self, path: Path, base_url: str, filenames: Iterable[str]):
        files = {}
        for filename in filenames:
            file_path = path.parent / filename
            if file_path.exists():
                files[filename] = {"size": file_path.stat().st_size}
        payload = {
            "base_url": base_url,
            "files": files,
        }
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
