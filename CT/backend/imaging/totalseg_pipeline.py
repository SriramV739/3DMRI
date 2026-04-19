from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from skimage import measure

from backend import settings


@dataclass(frozen=True)
class MeshStyle:
    color: tuple[int, int, int, int]
    step_size: int = 1
    smoothing_iterations: int = 8
    keep_largest_component: bool = False
    sdf_smoothing_mm: float = 1.2
    max_vertices: int | None = None


DEFAULT_STYLE = MeshStyle((206, 182, 145, 210), step_size=1, smoothing_iterations=8, sdf_smoothing_mm=1.4, max_vertices=45000)


def _style_for_label(label: str) -> MeshStyle:
    if label.startswith("lung_"):
        return MeshStyle((82, 170, 220, 120), smoothing_iterations=12, sdf_smoothing_mm=1.8, max_vertices=90000)
    if label in {"heart", "atrial_appendage_left"} or "heart" in label:
        return MeshStyle((218, 62, 70, 235), smoothing_iterations=14, keep_largest_component=True, sdf_smoothing_mm=1.4, max_vertices=120000)
    if "aorta" in label or "artery" in label or "trunk" in label:
        return MeshStyle((232, 55, 48, 240), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.0, max_vertices=60000)
    if "vena" in label or "vein" in label:
        return MeshStyle((60, 105, 224, 225), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.0, max_vertices=60000)
    if label in {"trachea", "esophagus"}:
        return MeshStyle((88, 218, 210, 220), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.0, max_vertices=50000)
    if label.startswith("rib_") or label.startswith("vertebrae_") or label in {
        "sternum",
        "clavicula_left",
        "clavicula_right",
        "scapula_left",
        "scapula_right",
        "humerus_left",
        "humerus_right",
        "costal_cartilages",
    }:
        return MeshStyle((238, 229, 201, 255), smoothing_iterations=8, sdf_smoothing_mm=0.9, max_vertices=70000)
    if label == "liver":
        return MeshStyle((154, 83, 68, 225), smoothing_iterations=14, keep_largest_component=True, sdf_smoothing_mm=1.8, max_vertices=120000)
    if label == "spleen":
        return MeshStyle((151, 92, 178, 220), smoothing_iterations=12, keep_largest_component=True, sdf_smoothing_mm=1.5, max_vertices=70000)
    if label.startswith("kidney"):
        return MeshStyle((209, 118, 64, 225), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.2, max_vertices=50000)
    if label in {"stomach", "small_bowel", "colon", "duodenum"}:
        return MeshStyle((218, 159, 88, 190), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.5, max_vertices=65000)
    if "muscle" in label or label.startswith(("autochthon", "iliopsoas", "gluteus")):
        return MeshStyle((142, 82, 72, 175), smoothing_iterations=8, sdf_smoothing_mm=1.5, max_vertices=55000)
    if label in {"brain", "spinal_cord"}:
        return MeshStyle((224, 179, 147, 215), smoothing_iterations=10, keep_largest_component=True, sdf_smoothing_mm=1.0, max_vertices=45000)
    return DEFAULT_STYLE


def _is_renderable_label(label: str) -> bool:
    skipped_exact = {
        "skin",
        "body",
        "body_trunc",
        "subcutaneous_fat",
        "torso_fat",
        "skeletal_muscle",
        "prostate",
        "urinary_bladder",
        "femur_left",
        "femur_right",
        "hip_left",
        "hip_right",
        "sacrum",
        "skull",
    }
    if label in skipped_exact:
        return False
    if label.startswith(("gluteus_", "iliac_")):
        return False
    return True


def _run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(settings.BASE_DIR),
            "TOTALSEG_HOME": str(settings.BASE_DIR / ".totalsegmentator"),
            "XDG_CACHE_HOME": str(settings.BASE_DIR / ".cache"),
            "HF_HOME": str(settings.BASE_DIR / ".cache" / "huggingface"),
            "TORCH_HOME": str(settings.BASE_DIR / ".cache" / "torch"),
            "nnUNet_raw": str(settings.BASE_DIR / ".cache" / "nnunet" / "raw"),
            "nnUNet_preprocessed": str(settings.BASE_DIR / ".cache" / "nnunet" / "preprocessed"),
            "nnUNet_results": str(settings.BASE_DIR / ".cache" / "nnunet" / "results"),
        }
    )
    for key in ("TOTALSEG_HOME", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def convert_nrrd_to_nifti(nrrd_path: Path = settings.RAW_CHEST_NRRD) -> Path:
    if not nrrd_path.exists():
        raise FileNotFoundError(f"Missing NRRD input: {nrrd_path}")
    settings.ensure_generated_dirs()
    output_path = settings.GENERATED_TOTALSEG_INPUTS_DIR / "CT-chest.nii.gz"
    image = sitk.ReadImage(str(nrrd_path))
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return output_path


def run_totalsegmentator(
    input_nifti: Path,
    output_dir: Path = settings.GENERATED_TOTALSEG_SEGMENTATIONS_DIR,
    device: str = "mps",
    fast: bool = True,
    force: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*.nii.gz"):
            path.unlink()
    if len(list(output_dir.glob("*.nii.gz"))) > 25:
        return output_dir

    command = [
        str(settings.BASE_DIR / ".venv" / "bin" / "TotalSegmentator"),
        "-i",
        str(input_nifti),
        "-o",
        str(output_dir),
        "-ot",
        "nifti",
        "-d",
        device,
        "-rmb",
        "-s",
    ]
    if fast:
        command.append("-f")

    try:
        subprocess.run(command, cwd=settings.BASE_DIR, env=_run_env(), check=True)
    except subprocess.CalledProcessError:
        if device != "cpu":
            cpu_command = [part if part != device else "cpu" for part in command]
            subprocess.run(cpu_command, cwd=settings.BASE_DIR, env=_run_env(), check=True)
        else:
            raise
    return output_dir


def _load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float], tuple[int, int, int]]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    mask = data > 0
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    return mask, zooms, tuple(int(value) for value in image.shape[:3])


def _crop_mask(mask: np.ndarray, pad: int = 3) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lower = np.maximum(coords.min(axis=0) - pad, 0)
    upper = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    cropped = mask[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    return cropped, lower


def _smooth_levelset(mask: np.ndarray, spacing: tuple[float, float, float], smooth_mm: float) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    levelset = inside - outside
    sigma = tuple(max(0.2, smooth_mm / value) for value in spacing)
    return ndimage.gaussian_filter(levelset.astype(np.float32), sigma=sigma)


def _decimate(mesh: trimesh.Trimesh, max_vertices: int | None) -> trimesh.Trimesh:
    if not max_vertices or len(mesh.vertices) <= max_vertices:
        return mesh
    target_faces = max(120, int(len(mesh.faces) * (max_vertices / len(mesh.vertices))))
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        return mesh


def _mask_to_mesh(mask: np.ndarray, zooms: tuple[float, float, float], style: MeshStyle) -> trimesh.Trimesh | None:
    if int(mask.sum()) < 16:
        return None
    if style.keep_largest_component:
        labels, count = ndimage.label(mask)
        if count > 1:
            sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
            largest = int(np.argmax(sizes) + 1)
            mask = labels == largest

    crop = _crop_mask(mask, pad=5)
    if crop is None:
        return None
    cropped, lower = crop

    volume_mask = np.pad(np.transpose(cropped, (2, 1, 0)), 3)
    spacing = (zooms[2], zooms[1], zooms[0])
    target_spacing = min(spacing)
    zoom_factors = tuple(max(1.0, value / target_spacing) for value in spacing)
    if any(factor > 1.05 for factor in zoom_factors):
        volume_mask = ndimage.zoom(volume_mask.astype(np.float32), zoom=zoom_factors, order=1) >= 0.5
        spacing = tuple(value / factor for value, factor in zip(spacing, zoom_factors, strict=True))

    levelset = _smooth_levelset(volume_mask, spacing, style.sdf_smoothing_mm)
    try:
        verts, faces, _normals, _values = measure.marching_cubes(
            levelset,
            level=0.0,
            spacing=spacing,
            step_size=style.step_size,
            allow_degenerate=False,
        )
    except ValueError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    verts = verts[:, [2, 1, 0]]
    verts -= np.array([spacing[2], spacing[1], spacing[0]]) * 3.0
    verts += np.array([lower[0] * zooms[0], lower[1] * zooms[1], lower[2] * zooms[2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh = _decimate(mesh, style.max_vertices)
    if style.smoothing_iterations:
        trimesh.smoothing.filter_humphrey(mesh, alpha=0.08, beta=0.55, iterations=style.smoothing_iterations)
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.18, iterations=max(1, style.smoothing_iterations // 3))
    mesh.visual.face_colors = np.tile(np.array(style.color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def _scene_transform(meshes: list[trimesh.Trimesh]) -> None:
    bounds = np.array([mesh.bounds for mesh in meshes if len(mesh.vertices) > 0])
    global_min = bounds[:, 0, :].min(axis=0)
    global_max = bounds[:, 1, :].max(axis=0)
    center = (global_min + global_max) / 2.0
    extent = float((global_max - global_min).max()) or 1.0
    scale = 4.8 / extent
    for mesh in meshes:
        mesh.vertices = (mesh.vertices - center) * scale


def build_totalseg_meshes(
    segmentation_dir: Path = settings.GENERATED_TOTALSEG_SEGMENTATIONS_DIR,
    output_id: str = "totalseg_CT_chest_realistic",
) -> dict[str, Any]:
    settings.ensure_generated_dirs()
    mask_paths = sorted(path for path in segmentation_dir.glob("*.nii.gz") if _is_renderable_label(path.name.removesuffix(".nii.gz")))
    if not mask_paths:
        raise FileNotFoundError(f"No TotalSegmentator masks found in {segmentation_dir}")

    scene = trimesh.Scene()
    meshes: list[trimesh.Trimesh] = []
    anatomy: list[dict[str, Any]] = []
    reference_shape: tuple[int, int, int] | None = None
    reference_zooms: tuple[float, float, float] | None = None

    for path in mask_paths:
        label = path.name.removesuffix(".nii.gz")
        style = _style_for_label(label)
        mask, zooms, shape = _load_mask(path)
        reference_shape = reference_shape or shape
        reference_zooms = reference_zooms or zooms
        mesh = _mask_to_mesh(mask, zooms, style)
        if mesh is None:
            continue
        mesh.metadata["name"] = label
        meshes.append(mesh)
        anatomy.append(
            {
                "label": label,
                "voxels": int(mask.sum()),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "color": list(style.color),
            }
        )

    if not meshes:
        raise RuntimeError("TotalSegmentator masks were present, but no renderable meshes were produced")

    _scene_transform(meshes)
    for mesh, item in zip(meshes, anatomy, strict=False):
        scene.add_geometry(mesh, node_name=item["label"], geom_name=item["label"])

    glb_path = settings.GENERATED_VOLUMES_DIR / f"{output_id}.glb"
    scene.export(glb_path, file_type="glb")

    manifest = {
        "id": output_id,
        "kind": "totalsegmentator",
        "volume_url": f"/generated/volumes/{glb_path.name}",
        "source": str(settings.RAW_CHEST_NRRD.relative_to(settings.BASE_DIR)),
        "converted_input": str((settings.GENERATED_TOTALSEG_INPUTS_DIR / "CT-chest.nii.gz").relative_to(settings.BASE_DIR)),
        "segmentation_dir": str(segmentation_dir.relative_to(settings.BASE_DIR)),
        "shape": list(reference_shape or []),
        "spacing": list(reference_zooms or []),
        "anatomy_count": len(anatomy),
        "mesh_vertices": int(sum(item["vertices"] for item in anatomy)),
        "mesh_faces": int(sum(item["faces"] for item in anatomy)),
        "anatomy": anatomy,
    }
    manifest_path = settings.GENERATED_VOLUMES_DIR / f"{output_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_totalseg_chest(
    run_segmentation: bool = True,
    force_segmentation: bool = False,
    device: str = "mps",
    fast: bool = True,
) -> dict[str, Any]:
    input_nifti = convert_nrrd_to_nifti()
    segmentation_dir = settings.GENERATED_TOTALSEG_SEGMENTATIONS_DIR
    if run_segmentation:
        segmentation_dir = run_totalsegmentator(
            input_nifti=input_nifti,
            output_dir=settings.GENERATED_TOTALSEG_SEGMENTATIONS_DIR,
            device=device,
            fast=fast,
            force=force_segmentation,
        )
    return build_totalseg_meshes(segmentation_dir=segmentation_dir)
