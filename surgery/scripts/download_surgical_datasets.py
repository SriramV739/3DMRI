"""
Download and extract the surgical datasets used for dual-YOLO training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List

import requests


DATASETS = {
    "m2cai16_tool_locations": {
        "files": [
            {
                "url": "https://ai.stanford.edu/~syyeung/resources/m2cai16-tool-locations.zip",
                "filename": "m2cai16-tool-locations.zip",
                "extract": "zip",
            }
        ],
    },
    "endoscapes": {
        "files": [
            {
                "url": "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip",
                "filename": "endoscapes.zip",
                "extract": "zip",
            }
        ],
    },
    "cholecseg8k": {
        "files": [
            {
                "url": "https://www.kaggle.com/api/v1/datasets/download/newslab/cholecseg8k",
                "filename": "cholecseg8k.zip",
                "extract": "zip",
            }
        ],
    },
    "cholec80_boxes": {
        "files": [
            {
                "url": "https://zenodo.org/api/records/13170928/files/ROI_Labels.csv/content",
                "filename": "ROI_Labels.csv",
                "extract": None,
            },
            {
                "url": "https://zenodo.org/api/records/13170928/files/Images.rar/content",
                "filename": "Images.rar",
                "extract": "rar",
            },
        ],
    },
}


def _download(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"  {output_path.name}: {pct:5.1f}% ({downloaded}/{total} bytes)")
                else:
                    print(f"  {output_path.name}: {downloaded} bytes")
    return output_path


def _extract_zip(archive_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)


def _extract_rar(archive_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_exe = shutil.which("tar")
    if tar_exe:
        result = subprocess.run(
            [tar_exe, "-xf", str(archive_path), "-C", str(output_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

    raise RuntimeError(
        "RAR extraction failed. Install a tool that can extract .rar archives, "
        "such as bsdtar with RAR support or 7-Zip."
    )


def download_dataset(dataset_name: str, root_dir: Path, extract: bool) -> Dict[str, object]:
    dataset_dir = root_dir / dataset_name
    downloads_dir = dataset_dir / "downloads"
    raw_dir = dataset_dir / "raw"
    manifest: Dict[str, object] = {
        "dataset": dataset_name,
        "downloads": [],
        "raw_dir": str(raw_dir),
    }

    for file_spec in DATASETS[dataset_name]["files"]:
        archive_path = downloads_dir / file_spec["filename"]
        if archive_path.exists():
            print(f"[skip] {archive_path} already exists")
        else:
            print(f"[download] {dataset_name}: {file_spec['filename']}")
            _download(file_spec["url"], archive_path)

        file_manifest = {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "extract": file_spec["extract"],
        }
        manifest["downloads"].append(file_manifest)

        if not extract or not file_spec["extract"]:
            continue

        if dataset_name == "cholec80_boxes" and file_spec["filename"] == "Images.rar":
            extract_dir = raw_dir
        else:
            extract_dir = raw_dir

        marker = extract_dir / f".{archive_path.stem}.extracted"
        if marker.exists():
            print(f"[skip] {archive_path.name} already extracted")
            continue

        print(f"[extract] {archive_path.name} -> {extract_dir}")
        if file_spec["extract"] == "zip":
            _extract_zip(archive_path, extract_dir)
        elif file_spec["extract"] == "rar":
            _extract_rar(archive_path, extract_dir)
        else:
            raise ValueError(f"Unsupported extract mode: {file_spec['extract']}")
        marker.write_text("ok", encoding="utf-8")

    manifest_path = dataset_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download the surgical datasets used by the dual-YOLO training pipeline"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to download",
    )
    parser.add_argument(
        "--root-dir",
        default="data/datasets",
        help="Root directory that will contain per-dataset downloads and raw extracts",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives only",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    summaries: List[Dict[str, object]] = []
    for dataset_name in args.datasets:
        summaries.append(download_dataset(dataset_name, root_dir, extract=not args.no_extract))

    summary_path = root_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {summary_path}")


if __name__ == "__main__":
    main()
