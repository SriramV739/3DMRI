from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.imaging.totalseg_pipeline import generate_totalseg_chest


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a TotalSegmentator anatomy GLB from data/raw/CT-chest.nrrd")
    parser.add_argument("--skip-segmentation", action="store_true", help="Only rebuild meshes from generated/totalseg/segmentations")
    parser.add_argument("--force-segmentation", action="store_true", help="Delete generated masks and rerun TotalSegmentator")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "gpu"], help="TotalSegmentator inference device")
    parser.add_argument("--fullres", action="store_true", help="Use TotalSegmentator's full-resolution model instead of --fast")
    args = parser.parse_args()

    manifest = generate_totalseg_chest(
        run_segmentation=not args.skip_segmentation,
        force_segmentation=args.force_segmentation,
        device=args.device,
        fast=not args.fullres,
    )
    print(
        "TOTALSEG_OK",
        manifest["id"],
        f"anatomy={manifest['anatomy_count']}",
        f"vertices={manifest['mesh_vertices']}",
        f"faces={manifest['mesh_faces']}",
        manifest["volume_url"],
    )


if __name__ == "__main__":
    main()
