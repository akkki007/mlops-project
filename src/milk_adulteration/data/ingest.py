"""DVC stage `ingest`: download the raw dataset and verify its checksum."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import requests

from milk_adulteration.config import load_params, resolve

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256 and sha256_of(dest) == expected_sha256:
        log.info("%s already present with matching checksum", dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    actual = sha256_of(tmp)
    if expected_sha256 and actual != expected_sha256:
        tmp.unlink()
        raise ValueError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}. "
            "The upstream file changed; review it and update ingest.sha256 in params.yaml."
        )
    tmp.replace(dest)
    log.info("Downloaded %s -> %s (sha256=%s)", url, dest, actual)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = load_params()["ingest"]
    download(p["url"], resolve(p["output"]), p.get("sha256"))


if __name__ == "__main__":
    main()
