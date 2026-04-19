from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.imaging.dicom_io import discover_slices
from backend.imaging.volume_pipeline import export_volume


def main() -> None:
    records = discover_slices(limit=1)
    if not records:
        raise SystemExit("No DICOM slices found in CTImages/dicom_dir")
    manifest = export_volume(records, output_id=f"smoke_{records[0].slice_id}", max_size=144)
    print("SMOKE_OK", manifest["id"], manifest["shape"], manifest["mesh_vertices"], manifest["mesh_faces"])


if __name__ == "__main__":
    main()
