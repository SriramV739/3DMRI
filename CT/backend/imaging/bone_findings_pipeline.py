from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from backend import settings


BONE_PREFIXES = ("rib_", "vertebrae_")
BONE_LABELS = {
    "sternum",
    "clavicula_left",
    "clavicula_right",
    "scapula_left",
    "scapula_right",
    "humerus_left",
    "humerus_right",
    "costal_cartilages",
}
DEFAULT_SOURCE_ID = "totalseg_CT_chest_realistic"
DEFAULT_MIN_CONFIDENCE = 0.55


@dataclass(frozen=True)
class FindingCandidate:
    bone_label: str
    finding_type: str
    confidence: float
    center_voxel: tuple[int, int, int]
    center_world_mm: tuple[float, float, float]
    bbox_voxel: tuple[tuple[int, int, int], tuple[int, int, int]]
    endpoint_voxels: tuple[tuple[int, int, int], tuple[int, int, int]]
    evidence: dict[str, Any]


def is_bone_label(label: str) -> bool:
    return label in BONE_LABELS or label.startswith(BONE_PREFIXES)


def findings_dir_for_source(source_id: str) -> Path:
    return settings.GENERATED_FINDINGS_DIR / source_id


def findings_manifest_path(source_id: str) -> Path:
    return findings_dir_for_source(source_id) / "findings.json"


def load_bone_findings_manifest(source_id: str = DEFAULT_SOURCE_ID) -> dict[str, Any]:
    path = findings_manifest_path(source_id)
    if not path.exists():
        raise FileNotFoundError(f"No bone findings manifest exists for {source_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_manifest_path(source_id: str) -> Path:
    return settings.GENERATED_VOLUMES_DIR / f"{source_id}.json"


def _load_source_manifest(source_id: str) -> dict[str, Any]:
    path = _source_manifest_path(source_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing source volume manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ct_nifti(source_manifest: dict[str, Any]) -> tuple[np.ndarray, tuple[float, float, float]]:
    relative = source_manifest.get("converted_input")
    candidate = settings.BASE_DIR / relative if isinstance(relative, str) else settings.GENERATED_TOTALSEG_INPUTS_DIR / "CT-chest.nii.gz"
    if not candidate.exists():
        if settings.RAW_CHEST_NRRD.exists():
            settings.GENERATED_TOTALSEG_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
            image = sitk.ReadImage(str(settings.RAW_CHEST_NRRD))
            candidate = settings.GENERATED_TOTALSEG_INPUTS_DIR / "CT-chest.nii.gz"
            sitk.WriteImage(image, str(candidate), useCompression=True)
        else:
            raise FileNotFoundError(f"Missing CT input for bone findings: {candidate}")
    image = nib.load(str(candidate))
    data = np.asanyarray(image.dataobj).astype(np.float32)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, spacing


def _load_mask(mask_path: Path) -> np.ndarray:
    image = nib.load(str(mask_path))
    return np.asanyarray(image.dataobj) > 0


def _mask_paths_for_source(source_manifest: dict[str, Any]) -> list[Path]:
    relative = source_manifest.get("segmentation_dir")
    segmentation_dir = settings.BASE_DIR / relative if isinstance(relative, str) else settings.GENERATED_TOTALSEG_SEGMENTATIONS_DIR
    if not segmentation_dir.exists():
        raise FileNotFoundError(f"Missing TotalSegmentator segmentation dir: {segmentation_dir}")
    return sorted(path for path in segmentation_dir.glob("*.nii.gz") if is_bone_label(path.name.removesuffix(".nii.gz")))


def _remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:][np.asarray(sizes) >= min_voxels] = True
    return keep[labels]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    order = np.argsort(np.asarray(sizes))[::-1]
    return [np.argwhere(labels == int(index + 1)) for index in order]


def _closest_component_points(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    points_a = coords_a.astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    points_b = coords_b.astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    tree = cKDTree(points_b)
    distances, indices = tree.query(points_a, k=1)
    source_index = int(np.argmin(distances))
    target_index = int(indices[source_index])
    return coords_a[source_index], coords_b[target_index], float(distances[source_index])


def _bbox_around_points(points: list[np.ndarray], shape: tuple[int, int, int], pad: int = 5) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    stacked = np.vstack(points)
    lower = np.maximum(stacked.min(axis=0) - pad, 0)
    upper = np.minimum(stacked.max(axis=0) + pad + 1, np.asarray(shape))
    return tuple(int(value) for value in lower), tuple(int(value) for value in upper)


def _finding_type(confidence: float, fragment_ratio: float, gap_mm: float) -> str:
    if fragment_ratio < 0.28 and gap_mm > 4.0:
        return "fragment_candidate"
    if confidence >= 0.68:
        return "fracture_candidate"
    return "deformation_candidate"


def detect_bone_findings_from_arrays(
    ct_hu: np.ndarray,
    bone_masks: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    cortical_hu_threshold: float = 450.0,
    fallback_hu_threshold: float = 300.0,
) -> list[FindingCandidate]:
    findings: list[FindingCandidate] = []
    if ct_hu.ndim != 3:
        raise ValueError("ct_hu must be a 3D array")

    for bone_label, mask in sorted(bone_masks.items()):
        if mask.shape != ct_hu.shape:
            raise ValueError(f"Mask shape for {bone_label} does not match CT shape: {mask.shape} vs {ct_hu.shape}")
        if int(mask.sum()) < 80:
            continue

        masked_hu = ct_hu[mask]
        threshold = cortical_hu_threshold
        cortical = mask & (ct_hu >= threshold)
        if int(cortical.sum()) < max(40, int(mask.sum() * 0.02)):
            threshold = fallback_hu_threshold
            cortical = mask & (ct_hu >= threshold)
        if int(cortical.sum()) < 40:
            continue

        min_component_voxels = max(18, int(cortical.sum() * 0.015))
        cleaned = _remove_small_components(cortical, min_component_voxels)
        components = _component_coords(cleaned)
        if len(components) < 2:
            continue

        primary, secondary = components[0], components[1]
        primary_size = int(len(primary))
        secondary_size = int(len(secondary))
        total_size = int(sum(len(component) for component in components))
        fragment_ratio = secondary_size / max(primary_size, 1)
        if secondary_size < 24 or fragment_ratio < 0.08:
            continue

        endpoint_a, endpoint_b, gap_mm = _closest_component_points(primary, secondary, spacing)
        centroid_a = primary.mean(axis=0)
        centroid_b = secondary.mean(axis=0)
        displacement_mm = float(np.linalg.norm((centroid_a - centroid_b) * np.asarray(spacing)))
        extra_fragments = max(0, len(components) - 2)
        hu_mean = float(np.mean(masked_hu))
        hu_p95 = float(np.percentile(masked_hu, 95))

        gap_score = min(0.26, gap_mm / 28.0)
        fragment_score = min(0.20, fragment_ratio * 0.42)
        displacement_score = min(0.18, displacement_mm / 190.0)
        roughness_score = min(0.12, extra_fragments * 0.025)
        rib_bonus = 0.08 if bone_label.startswith("rib_") else 0.03
        confidence = min(0.96, 0.22 + gap_score + fragment_score + displacement_score + roughness_score + rib_bonus)
        if confidence < min_confidence:
            continue

        center = np.round((endpoint_a + endpoint_b) / 2.0).astype(int)
        bbox = _bbox_around_points([endpoint_a, endpoint_b], ct_hu.shape, pad=6)
        finding_type = _finding_type(confidence, fragment_ratio, gap_mm)
        findings.append(
            FindingCandidate(
                bone_label=bone_label,
                finding_type=finding_type,
                confidence=round(float(confidence), 3),
                center_voxel=tuple(int(value) for value in center),
                center_world_mm=tuple(round(float(value), 3) for value in center * np.asarray(spacing)),
                bbox_voxel=bbox,
                endpoint_voxels=(
                    tuple(int(value) for value in endpoint_a),
                    tuple(int(value) for value in endpoint_b),
                ),
                evidence={
                    "threshold_hu": threshold,
                    "gap_mm": round(gap_mm, 3),
                    "component_count": len(components),
                    "primary_component_voxels": primary_size,
                    "secondary_component_voxels": secondary_size,
                    "total_cortical_component_voxels": total_size,
                    "secondary_to_primary_ratio": round(float(fragment_ratio), 3),
                    "centroid_displacement_mm": round(displacement_mm, 3),
                    "masked_hu_mean": round(hu_mean, 2),
                    "masked_hu_p95": round(hu_p95, 2),
                },
            )
        )

    return findings


def _overlay_mask_for_finding(shape: tuple[int, int, int], finding: FindingCandidate) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for point in finding.endpoint_voxels:
        zyx = tuple(slice(max(0, value - 1), min(shape[index], value + 2)) for index, value in enumerate(point))
        mask[zyx] = True
    return ndimage.binary_dilation(mask, iterations=4)


def _mask_to_mesh(mask: np.ndarray, spacing: tuple[float, float, float], color: tuple[int, int, int, int]) -> trimesh.Trimesh | None:
    if int(mask.sum()) < 8:
        return None
    coords = np.argwhere(mask)
    lower = np.maximum(coords.min(axis=0) - 3, 0)
    upper = np.minimum(coords.max(axis=0) + 4, np.asarray(mask.shape))
    cropped = mask[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    volume_mask = np.pad(np.transpose(cropped, (2, 1, 0)).astype(np.float32), 2)
    try:
        verts, faces, _normals, _values = measure.marching_cubes(volume_mask, level=0.5, spacing=(spacing[2], spacing[1], spacing[0]))
    except ValueError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = verts[:, [2, 1, 0]]
    verts -= np.array([spacing[0], spacing[1], spacing[2]]) * 2.0
    verts += np.array([lower[0] * spacing[0], lower[1] * spacing[1], lower[2] * spacing[2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.25, iterations=2)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def _color_for_finding(confidence: float) -> tuple[int, int, int, int]:
    if confidence >= 0.70:
        return (255, 82, 54, 230)
    return (255, 178, 54, 218)


def _label_for_finding(finding: FindingCandidate) -> str:
    readable = finding.bone_label.replace("_", " ").title()
    if finding.finding_type == "fracture_candidate":
        return f"{readable} fracture candidate"
    if finding.finding_type == "fragment_candidate":
        return f"{readable} fragment candidate"
    return f"{readable} deformation candidate"


def run_bone_findings(
    source_id: str = DEFAULT_SOURCE_ID,
    force: bool = False,
    bone_labels: list[str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    settings.ensure_generated_dirs()
    output_dir = findings_dir_for_source(source_id)
    manifest_path = output_dir / "findings.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load_source_manifest(source_id)
    ct_hu, spacing = _read_ct_nifti(source_manifest)
    allowed = set(bone_labels or [])
    bone_masks: dict[str, np.ndarray] = {}
    for path in _mask_paths_for_source(source_manifest):
        label = path.name.removesuffix(".nii.gz")
        if allowed and label not in allowed:
            continue
        mask = _load_mask(path)
        if mask.shape != ct_hu.shape:
            raise ValueError(f"Mask shape for {label} does not match CT shape: {mask.shape} vs {ct_hu.shape}")
        bone_masks[label] = mask

    candidates = detect_bone_findings_from_arrays(ct_hu, bone_masks, spacing, min_confidence=min_confidence)
    findings: list[dict[str, Any]] = []
    overlay_scene = trimesh.Scene()
    for index, candidate in enumerate(candidates, start=1):
        finding_id = f"finding_{candidate.bone_label}_{index:03d}"
        mesh_label = finding_id
        color = _color_for_finding(candidate.confidence)
        overlay_mask = _overlay_mask_for_finding(ct_hu.shape, candidate)
        mesh = _mask_to_mesh(overlay_mask, spacing, color)
        mesh_path = output_dir / f"{mesh_label}.glb"
        if mesh is not None:
            mesh.metadata["name"] = mesh_label
            trimesh.Scene({mesh_label: mesh}).export(mesh_path, file_type="glb")
            overlay_scene.add_geometry(mesh.copy(), node_name=mesh_label, geom_name=mesh_label)

        findings.append(
            {
                "id": finding_id,
                "type": candidate.finding_type,
                "label": _label_for_finding(candidate),
                "bone_label": candidate.bone_label,
                "confidence": candidate.confidence,
                "mesh_label": mesh_label,
                "mesh_path": str(mesh_path.relative_to(settings.BASE_DIR)) if mesh_path.exists() else None,
                "color": list(color),
                "center_voxel": list(candidate.center_voxel),
                "center_world_mm": list(candidate.center_world_mm),
                "bbox_voxel": [list(candidate.bbox_voxel[0]), list(candidate.bbox_voxel[1])],
                "evidence": candidate.evidence,
                "review_required": True,
            }
        )

    overlay_glb_path = output_dir / "bone_findings_overlays.glb"
    if overlay_scene.geometry:
        overlay_scene.export(overlay_glb_path, file_type="glb")

    manifest = {
        "id": f"{source_id}_bone_findings",
        "kind": "bone_findings",
        "source_volume_id": source_id,
        "ct_only": True,
        "mri_supported": False,
        "min_confidence": min_confidence,
        "bone_labels_requested": sorted(allowed),
        "bone_labels_evaluated": sorted(bone_masks),
        "spacing": list(spacing),
        "finding_count": len(findings),
        "findings": findings,
        "overlay_glb": str(overlay_glb_path.relative_to(settings.BASE_DIR)) if overlay_glb_path.exists() else None,
        "notes": [
            "Deterministic CT HU/geometry candidate detector.",
            "Findings are prototype candidates and require radiologist review.",
            "Gemini is not used as the primary detector.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
