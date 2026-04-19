from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.imaging.volume_pipeline import generate_batch


def main() -> None:
    manifests = generate_batch(limit=10, max_size=144)
    if len(manifests) != 10:
        raise SystemExit(f"Expected 10 renders, generated {len(manifests)}")
    for manifest in manifests:
        print("BATCH_OK", manifest["id"], manifest["shape"], manifest["mesh_vertices"], manifest["mesh_faces"])


if __name__ == "__main__":
    main()
