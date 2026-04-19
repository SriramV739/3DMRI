"""
Fine-tune a single YOLO model on the Roboflow laparoscopy dataset for POC use.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train a single POC YOLO detector on the Roboflow laparoscopy dataset"
    )
    parser.add_argument(
        "--data",
        default="data/datasets/roboflow_laparoscopy/data.yaml",
        help="Path to the Roboflow-exported YOLO dataset yaml",
    )
    parser.add_argument("--base-model", default="yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project-dir", default="runs/train")
    parser.add_argument("--run-name", default="laparoscopy_poc")
    parser.add_argument("--output", default="models/laparoscopy_poc.pt")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project_dir,
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        amp=True,
    )

    best = Path(args.project_dir) / args.run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Expected trained weights at {best}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"[ok] Copied {best} -> {output}")


if __name__ == "__main__":
    main()
