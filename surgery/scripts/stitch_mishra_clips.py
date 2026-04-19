"""Stitch Dr. R. K. Mishra converted clips into one replayer asset.

This script expects the per-clip converted frame folders to already exist under
``surgery/data/converted``. It creates a new converted asset with sequential
JPEG frames and an MP4 convenience export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2


def stitch_mishra_clips(
    *,
    converted_root: Path,
    start_clip: int = 2,
    end_clip: int = 11,
    fps: int = 25,
    output_name: str | None = None,
) -> dict:
    converted_root = converted_root.resolve()
    clip_names = [
        f"dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_clip_{idx:03d}"
        for idx in range(start_clip, end_clip + 1)
    ]
    output_name = output_name or (
        "dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_"
        f"clips_{start_clip:03d}_{end_clip:03d}_stitched"
    )
    output_dir = converted_root / output_name
    frames_dir = output_dir / "frames"
    mp4_path = output_dir / f"{output_name}.mp4"
    metadata_path = output_dir / f"{output_name}_metadata.json"
    manifest_path = output_dir / f"{output_name}_source_manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        resolved = frames_dir.resolve()
        if converted_root not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    stitched_idx = 0
    width = 0
    height = 0
    for clip_name in clip_names:
        clip_dir = converted_root / clip_name
        src_frames = sorted((clip_dir / "frames").glob("*.jpg"))
        if not src_frames:
            raise RuntimeError(f"No frames found for {clip_name}")

        meta_path = clip_dir / f"{clip_name}_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        start_idx = stitched_idx
        for src in src_frames:
            dst = frames_dir / f"{stitched_idx:05d}.jpg"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            stitched_idx += 1

        if width == 0 or height == 0:
            sample = cv2.imread(str(src_frames[0]))
            if sample is None:
                raise RuntimeError(f"Could not read sample frame: {src_frames[0]}")
            height, width = sample.shape[:2]

        sources.append(
            {
                "clip": clip_name,
                "source_dir": str(clip_dir),
                "source_frame_count": len(src_frames),
                "stitched_start_frame": start_idx,
                "stitched_end_frame": stitched_idx - 1,
                "metadata_frame_count": meta.get("frame_count"),
                "resolution": meta.get("resolution"),
            }
        )

    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4: {mp4_path}")
    try:
        for idx in range(stitched_idx):
            frame = cv2.imread(str(frames_dir / f"{idx:05d}.jpg"))
            if frame is None:
                raise RuntimeError(f"Could not read stitched frame {idx}")
            writer.write(frame)
    finally:
        writer.release()

    metadata = {
        "name": output_name,
        "source_path": f"stitched from converted clips {start_clip:03d}-{end_clip:03d}",
        "output_dir": str(output_dir),
        "bundle_path": None,
        "frames_dir": str(frames_dir),
        "metadata_path": str(metadata_path),
        "frame_count": stitched_idx,
        "fps": fps,
        "resolution": [width, height],
        "format": "uint8_rgb",
        "stitched_clips": clip_names,
        "mp4_path": str(mp4_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", default="surgery/data/converted")
    parser.add_argument("--start-clip", type=int, default=2)
    parser.add_argument("--end-clip", type=int, default=11)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    metadata = stitch_mishra_clips(
        converted_root=Path(args.converted_root),
        start_clip=args.start_clip,
        end_clip=args.end_clip,
        fps=args.fps,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
