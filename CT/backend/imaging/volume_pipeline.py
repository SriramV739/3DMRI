from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import zoom
from skimage import measure

from backend import settings
from backend.imaging.dicom_io import (
    SliceRecord,
    discover_slices,
    read_hu_slice,
    save_colored_slice_png,
)


ISO_LEVELS = [
    {"name": "lung_air_boundary", "level": -550.0, "color": [72, 147, 190, 135]},
    {"name": "soft_tissue", "level": 45.0, "color": [220, 118, 112, 195]},
    {"name": "contrast_and_bone", "level": 300.0, "color": [238, 205, 117, 225]},
    {"name": "cortical_bone", "level": 650.0, "color": [247, 244, 221, 255]},
]


def _sort_records(records: list[SliceRecord]) -> list[SliceRecord]:
    def key(record: SliceRecord) -> tuple[float, float, str]:
        location = record.slice_location if record.slice_location is not None else 0.0
        instance = record.instance_number if record.instance_number is not None else 0.0
        return (location, instance, record.slice_id)

    return sorted(records, key=key)


def choose_render_groups(limit: int) -> list[list[SliceRecord]]:
    records = discover_slices()
    by_series: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        by_series[record.series_uid or record.slice_id].append(record)

    multi = [_sort_records(group) for group in by_series.values() if len(group) > 2]
    single_or_pair = [[record] for record in records]
    groups = multi + single_or_pair
    return groups[:limit]


def _downsample_slice(hu: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_size / float(max(hu.shape)))
    if scale == 1.0:
        return hu.astype(np.float32), 1.0
    resized = zoom(hu, zoom=(scale, scale), order=1)
    return resized.astype(np.float32), scale


def _single_slice_slab(hu: np.ndarray, depth: int = 18) -> np.ndarray:
    air = np.full_like(hu, -1024, dtype=np.float32)
    slab = np.repeat(hu[np.newaxis, :, :], depth, axis=0)
    return np.concatenate([air[np.newaxis, :, :], slab, air[np.newaxis, :, :]], axis=0)


def _stack_records(records: list[SliceRecord], max_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    slices: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for record in _sort_records(records):
        hu, meta = read_hu_slice(record.dicom_path)
        hu_small, scale = _downsample_slice(hu, max_size=max_size)
        slices.append(hu_small)
        meta["downsample_scale"] = scale
        metas.append(meta)
        save_colored_slice_png(hu, settings.GENERATED_SLICES_DIR / f"{record.slice_id}.png")

    if len(slices) == 1:
        volume = _single_slice_slab(slices[0])
    else:
        volume = np.stack(slices, axis=0)
        pad = np.full_like(volume[:1], -1024, dtype=np.float32)
        volume = np.concatenate([pad, volume, pad], axis=0)

    base_spacing = metas[0]["spacing"]
    scale = metas[0]["downsample_scale"]
    spacing = (
        float(base_spacing[0]),
        float(base_spacing[1]) / scale,
        float(base_spacing[2]) / scale,
    )
    return volume.astype(np.float32), {"source_meta": metas, "spacing": spacing}


def _mesh_for_level(volume: np.ndarray, spacing: tuple[float, float, float], level: float, color: list[int]) -> trimesh.Trimesh | None:
    if not (float(np.nanmin(volume)) < level < float(np.nanmax(volume))):
        return None
    verts, faces, _normals, _values = measure.marching_cubes(
        volume,
        level=level,
        spacing=spacing,
        step_size=2,
        allow_degenerate=False,
    )
    if len(verts) == 0 or len(faces) == 0:
        return None

    verts = verts[:, [2, 1, 0]]
    verts -= verts.mean(axis=0, keepdims=True)
    max_extent = float(np.max(np.ptp(verts, axis=0))) or 1.0
    verts = verts / max_extent * 3.2

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(faces), 1))
    return mesh


def export_volume(records: list[SliceRecord], output_id: str, max_size: int = 176) -> dict[str, Any]:
    settings.ensure_generated_dirs()
    volume, volume_meta = _stack_records(records, max_size=max_size)
    spacing = volume_meta["spacing"]

    meshes: list[trimesh.Trimesh] = []
    emitted_levels: list[dict[str, Any]] = []
    for spec in ISO_LEVELS:
        mesh = _mesh_for_level(volume, spacing, spec["level"], spec["color"])
        if mesh is not None:
            meshes.append(mesh)
            emitted_levels.append(spec)

    if not meshes:
        raise RuntimeError(f"No isosurfaces could be generated for {output_id}")

    combined = trimesh.util.concatenate(meshes)
    glb_path = settings.GENERATED_VOLUMES_DIR / f"{output_id}.glb"
    combined.export(glb_path, file_type="glb")

    source_records = [asdict(record) for record in records]
    for record in source_records:
        record["dicom_path"] = str(Path(record["dicom_path"]).relative_to(settings.BASE_DIR))
        record["tiff_path"] = str(Path(record["tiff_path"]).relative_to(settings.BASE_DIR)) if record["tiff_path"] else None

    manifest = {
        "id": output_id,
        "volume_url": f"/generated/volumes/{glb_path.name}",
        "source_slice_ids": [record.slice_id for record in records],
        "source_records": source_records,
        "shape": list(volume.shape),
        "hu_min": float(np.nanmin(volume)),
        "hu_max": float(np.nanmax(volume)),
        "spacing": list(spacing),
        "surfaces": emitted_levels,
        "mesh_vertices": int(len(combined.vertices)),
        "mesh_faces": int(len(combined.faces)),
    }
    manifest_path = settings.GENERATED_VOLUMES_DIR / f"{output_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_batch(limit: int = 10, max_size: int = 176) -> list[dict[str, Any]]:
    manifests = []
    for index, records in enumerate(choose_render_groups(limit), start=1):
        source = records[0].slice_id
        output_id = f"ct_render_{index:02d}_{source}"
        manifests.append(export_volume(records, output_id=output_id, max_size=max_size))
    return manifests


def load_volume_manifests() -> list[dict[str, Any]]:
    settings.ensure_generated_dirs()
    manifests = []
    for path in sorted(settings.GENERATED_VOLUMES_DIR.glob("*.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(manifests, key=lambda item: (0 if item.get("kind") == "totalsegmentator" else 1, item.get("id", "")))
