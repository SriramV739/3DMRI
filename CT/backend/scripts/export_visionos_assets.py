from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.imaging.visionos_export import export_visionos_asset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Vision Pro optimized USDZ/GLB asset from a generated CT volume.")
    parser.add_argument("--source-id", default="totalseg_CT_chest_realistic", help="Generated volume id without extension.")
    parser.add_argument("--quality", choices=["preview", "balanced", "hq"], default="balanced")
    parser.add_argument("--force", action="store_true", help="Regenerate even when outputs already exist.")
    args = parser.parse_args()

    manifest = export_visionos_asset(source_id=args.source_id, quality_name=args.quality, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
