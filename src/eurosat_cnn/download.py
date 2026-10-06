"""Download and extract the EuroSAT RGB dataset (about 90 MB).

Source: Helber et al., EuroSAT, distributed via Zenodo (https://zenodo.org/records/7711810).

Example:
    python -m eurosat_cnn.download
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import CLASSES, DATA_DIR

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=URL)
    p.add_argument("--target", default=str(DATA_DIR))
    args = p.parse_args(argv)

    target = Path(args.target)
    if target.exists() and all((target / c).exists() for c in CLASSES):
        print(f"Dataset already present in {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target.parent / "EuroSAT_RGB.zip"

    print(f"Downloading {args.url} ...")
    with urllib.request.urlopen(args.url) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target.parent)
    # The archive contains a single top-level folder with one sub-folder per class.
    extracted = next((d for d in target.parent.iterdir() if d.is_dir() and (d / CLASSES[0]).exists()), None)
    if extracted is None:
        raise RuntimeError("Could not find the class folders in the downloaded archive.")
    if extracted != target:
        extracted.rename(target)
    zip_path.unlink()
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
