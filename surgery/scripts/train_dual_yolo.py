"""
Train the tool and anatomy YOLO models used by the MedSAM2 pipeline.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train_model(
    data_yaml: Path,
    base_model: str,
    run_name: str,
    project_dir: Path,
    output_weight: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    print(f"[train] {run_name} using {data_yaml}")
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        amp=True,
        cache=False,
    )

    best_weight = project_dir / run_name / "weights" / "best.pt"
    if not best_weight.exists():
        raise RuntimeError(f"Expected trained weights at {best_weight}")
    output_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, output_weight)
    print(f"[ok] Copied {best_weight} -> {output_weight}")


def main():
    parser = argparse.ArgumentParser(
        description="Train dual YOLO checkpoints for tools and anatomy"
    )
    parser.add_argument(
        "--tools-data",
        default="data/datasets/prepared/yolo_tools_dataset.yaml",
        help="YOLO dataset yaml for the tool detector",
    )
    parser.add_argument(
        "--anatomy-data",
        default="data/datasets/prepared/yolo_anatomy_dataset.yaml",
        help="YOLO dataset yaml for the anatomy detector",
    )
    parser.add_argument(
        "--tools-base-model",
        default="yolov8n.pt",
        help="Base YOLO checkpoint for the tool detector",
    )
    parser.add_argument(
        "--anatomy-base-model",
        default="yolov8n.pt",
        help="Base YOLO checkpoint for the anatomy detector",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for both runs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", default="0", help="Ultralytics device string")
    parser.add_argument("--workers", type=int, default=8, help="Data-loader workers")
    parser.add_argument(
        "--project-dir",
        default="runs/train",
        help="Ultralytics output directory",
    )
    parser.add_argument(
        "--tools-output",
        default="models/yolo_tools.pt",
        help="Final location for the trained tool weights",
    )
    parser.add_argument(
        "--anatomy-output",
        default="models/yolo_anatomy.pt",
        help="Final location for the trained anatomy weights",
    )
    parser.add_argument("--skip-tools", action="store_true", help="Skip training the tool model")
    parser.add_argument("--skip-anatomy", action="store_true", help="Skip training the anatomy model")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not args.skip_tools:
        train_model(
            data_yaml=Path(args.tools_data),
            base_model=args.tools_base_model,
            run_name="yolo_tools",
            project_dir=project_dir,
            output_weight=Path(args.tools_output),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )

    if not args.skip_anatomy:
        train_model(
            data_yaml=Path(args.anatomy_data),
            base_model=args.anatomy_base_model,
            run_name="yolo_anatomy",
            project_dir=project_dir,
            output_weight=Path(args.anatomy_output),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
