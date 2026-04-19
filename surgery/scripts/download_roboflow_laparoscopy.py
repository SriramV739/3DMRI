"""
Download the public Roboflow Universe laparoscopy dataset for local fine-tuning.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboflow import Roboflow


def main():
    parser = argparse.ArgumentParser(
        description="Download the Roboflow Universe laparoscopy dataset"
    )
    parser.add_argument("--workspace", default="laparoscopic-yolo")
    parser.add_argument("--project", default="laparoscopy")
    parser.add_argument("--version", type=int, default=14)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--output-dir", default="data/datasets/roboflow_laparoscopy")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Roboflow API key. Public Universe datasets may work without one.",
    )
    args = parser.parse_args()

    api_key = args.api_key if args.api_key is not None else os.getenv("ROBOFLOW_API_KEY", "")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format, location=str(Path(args.output_dir)))
    print(dataset)


if __name__ == "__main__":
    main()
