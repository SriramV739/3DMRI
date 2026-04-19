from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from backend import settings
from backend.imaging.bone_findings_pipeline import findings_manifest_path


@dataclass(frozen=True)
class VisionOSQuality:
    name: str
    target_vertices: int
    max_vertices_per_mesh: int
    min_vertices_per_mesh: int


QUALITY_PRESETS = {
    "preview": VisionOSQuality("preview", target_vertices=280_000, max_vertices_per_mesh=16_000, min_vertices_per_mesh=350),
    "balanced": VisionOSQuality("balanced", target_vertices=700_000, max_vertices_per_mesh=42_000, min_vertices_per_mesh=700),
    "hq": VisionOSQuality("hq", target_vertices=1_200_000, max_vertices_per_mesh=72_000, min_vertices_per_mesh=1_000),
}


def _source_manifest_path(source_id: str) -> Path:
    return settings.GENERATED_VOLUMES_DIR / f"{source_id}.json"


def _source_glb_path(source_id: str) -> Path:
    return settings.GENERATED_VOLUMES_DIR / f"{source_id}.glb"


def _safe_usd_identifier(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", label)
    if not cleaned:
        return "Anatomy"
    if cleaned[0].isdigit():
        return f"_{cleaned}"
    return cleaned


def _anatomy_group(label: str) -> str:
    if label.startswith("lung_"):
        return "lungs"
    if label == "heart" or "heart" in label or "atrial" in label or "ventricle" in label:
        return "heart"
    if "artery" in label or "aorta" in label or "trunk" in label:
        return "arteries"
    if "vein" in label or "vena" in label:
        return "veins"
    if (
        label.startswith("rib_")
        or label.startswith("vertebrae_")
        or label in {"sternum", "costal_cartilages"}
        or label.startswith(("clavicula_", "scapula_", "humerus_"))
    ):
        return "bones"
    if label in {
        "liver",
        "spleen",
        "stomach",
        "duodenum",
        "colon",
        "small_bowel",
        "kidney_left",
        "kidney_right",
        "pancreas",
        "gallbladder",
        "adrenal_gland_left",
        "adrenal_gland_right",
    }:
        return "abdomen"
    if label in {"trachea", "esophagus"}:
        return "airway"
    if "muscle" in label or label.startswith(("autochthon", "iliopsoas")):
        return "muscle"
    return "other"


def _load_source_manifest(source_id: str) -> dict[str, Any]:
    manifest_path = _source_manifest_path(source_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_scene(source_id: str) -> trimesh.Scene:
    glb_path = _source_glb_path(source_id)
    if not glb_path.exists():
        raise FileNotFoundError(f"Missing source GLB: {glb_path}")
    scene = trimesh.load(glb_path, force="scene")
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise RuntimeError(f"Could not load renderable scene from {glb_path}")
    return scene


def _color_for_label(label: str, anatomy_by_label: dict[str, dict[str, Any]]) -> tuple[float, float, float, float]:
    raw = anatomy_by_label.get(label, {}).get("color", [205, 195, 172, 230])
    rgba = [max(0, min(255, int(value))) for value in raw[:4]]
    if len(rgba) < 4:
        rgba.append(255)
    return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0)


def _weight_for_label(label: str) -> float:
    group = _anatomy_group(label)
    if group in {"heart", "lungs", "arteries", "veins"}:
        return 1.45
    if group == "airway":
        return 1.2
    if group == "bones":
        return 0.82
    if group == "muscle":
        return 0.65
    return 1.0


def _target_vertices(label: str, original_vertices: int, total_weighted_vertices: float, quality: VisionOSQuality) -> int:
    weighted_share = (original_vertices * _weight_for_label(label)) / max(total_weighted_vertices, 1.0)
    target = int(weighted_share * quality.target_vertices)
    target = max(quality.min_vertices_per_mesh, target)
    target = min(quality.max_vertices_per_mesh, target)
    return min(original_vertices, target)


def _simplify_mesh(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    working = mesh.copy()
    working.remove_unreferenced_vertices()
    if len(working.vertices) <= target_vertices:
        return working

    target_faces = max(120, int(len(working.faces) * (target_vertices / max(len(working.vertices), 1))))
    simplified = working.simplify_quadric_decimation(face_count=target_faces, aggression=4)
    if hasattr(simplified, "remove_degenerate_faces"):
        simplified.remove_degenerate_faces()
    else:
        simplified.update_faces(simplified.nondegenerate_faces())
    simplified.remove_unreferenced_vertices()
    simplified.merge_vertices()
    return simplified


def _make_material(stage: Usd.Stage, label: str, rgba: tuple[float, float, float, float]) -> UsdShade.Material:
    safe_name = _safe_usd_identifier(label)
    material = UsdShade.Material.Define(stage, f"/Materials/{safe_name}")
    shader = UsdShade.Shader.Define(stage, f"/Materials/{safe_name}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(rgba[0], rgba[1], rgba[2]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _write_usdz(
    meshes: list[tuple[str, trimesh.Trimesh, tuple[float, float, float, float]]],
    usdz_path: Path,
    finding_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    temp_dir = usdz_path.parent / f".{usdz_path.stem}_usd"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    usda_path = temp_dir / f"{usdz_path.stem}.usda"

    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/CTAnatomy")
    stage.SetDefaultPrim(root.GetPrim())

    finding_meta = finding_metadata or {}
    materials: dict[str, UsdShade.Material] = {}
    for label, mesh, rgba in meshes:
        safe_name = _safe_usd_identifier(label)
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/CTAnatomy/{safe_name}")
        points = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertices]
        faces = np.asarray(mesh.faces, dtype=np.int64)
        usd_mesh.CreatePointsAttr(points)
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.GetPrim().CreateAttribute("userProperties:anatomyLabel", Sdf.ValueTypeNames.String).Set(label)

        finding_info = finding_meta.get(label)
        if finding_info:
            usd_mesh.GetPrim().CreateAttribute("userProperties:findingId", Sdf.ValueTypeNames.String).Set(finding_info.get("id", ""))
            usd_mesh.GetPrim().CreateAttribute("userProperties:findingType", Sdf.ValueTypeNames.String).Set(finding_info.get("type", ""))

        display_color = usd_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        display_color.Set([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
        display_opacity = usd_mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant)
        display_opacity.Set([float(rgba[3])])

        material = materials.get(label)
        if material is None:
            material = _make_material(stage, label, rgba)
            materials[label] = material
        UsdShade.MaterialBindingAPI(usd_mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    usdz_path.unlink(missing_ok=True)
    if not UsdUtils.CreateNewUsdzPackage(str(usda_path), str(usdz_path)):
        raise RuntimeError(f"Failed to create USDZ package at {usdz_path}")
    shutil.rmtree(temp_dir)


def export_visionos_asset(source_id: str = "totalseg_CT_chest_realistic", quality_name: str = "balanced", force: bool = False) -> dict[str, Any]:
    settings.ensure_generated_dirs()
    quality = QUALITY_PRESETS.get(quality_name)
    if quality is None:
        raise ValueError(f"Unknown quality '{quality_name}'. Expected one of: {', '.join(QUALITY_PRESETS)}")

    output_id = f"{source_id}_visionos_{quality.name}"
    manifest_path = settings.GENERATED_VISIONOS_DIR / f"{output_id}.json"
    glb_path = settings.GENERATED_VISIONOS_DIR / f"{output_id}.glb"
    usdz_path = settings.GENERATED_VISIONOS_DIR / f"{output_id}.usdz"
    if not force and manifest_path.exists() and glb_path.exists() and usdz_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    source_manifest = _load_source_manifest(source_id)
    anatomy_by_label = {item["label"]: item for item in source_manifest.get("anatomy", []) if "label" in item}
    scene = _load_scene(source_id)

    total_weighted_vertices = sum(len(mesh.vertices) * _weight_for_label(label) for label, mesh in scene.geometry.items())
    optimized_scene = trimesh.Scene()
    optimized_meshes: list[tuple[str, trimesh.Trimesh, tuple[float, float, float, float]]] = []
    anatomy: list[dict[str, Any]] = []

    for label, mesh in sorted(scene.geometry.items()):
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        target = _target_vertices(label, len(mesh.vertices), total_weighted_vertices, quality)
        optimized = _simplify_mesh(mesh, target)
        rgba = _color_for_label(label, anatomy_by_label)
        optimized.visual.face_colors = np.tile(
            np.array([int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(rgba[3] * 255)], dtype=np.uint8),
            (len(optimized.faces), 1),
        )
        optimized_scene.add_geometry(optimized, node_name=label, geom_name=label)
        optimized_meshes.append((label, optimized, rgba))
        anatomy.append(
            {
                "label": label,
                "group": _anatomy_group(label),
                "color": [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(rgba[3] * 255)],
                "source_vertices": int(len(mesh.vertices)),
                "vertices": int(len(optimized.vertices)),
                "faces": int(len(optimized.faces)),
            }
        )

    if not optimized_meshes:
        raise RuntimeError("No meshes were available for Vision Pro export")

    # --- Load and merge bone findings if they exist ---
    findings_items: list[dict[str, Any]] = []
    finding_meta_for_usdz: dict[str, dict[str, Any]] = {}
    findings_path = findings_manifest_path(source_id)
    if findings_path.exists():
        try:
            findings_data = json.loads(findings_path.read_text(encoding="utf-8"))
            for finding in findings_data.get("findings", []):
                mesh_label = finding.get("mesh_label", "")
                finding_glb = settings.GENERATED_FINDINGS_DIR / source_id / f"{mesh_label}.glb"
                if not finding_glb.exists() or not mesh_label:
                    continue
                try:
                    finding_scene = trimesh.load(finding_glb, force="scene")
                    if isinstance(finding_scene, trimesh.Scene) and finding_scene.geometry:
                        for _name, finding_mesh in finding_scene.geometry.items():
                            if not isinstance(finding_mesh, trimesh.Trimesh) or len(finding_mesh.vertices) == 0:
                                continue
                            raw_color = finding.get("color", [255, 82, 54, 230])
                            rgba = (raw_color[0] / 255.0, raw_color[1] / 255.0, raw_color[2] / 255.0, raw_color[3] / 255.0 if len(raw_color) > 3 else 0.9)
                            finding_mesh.visual.face_colors = np.tile(
                                np.array(raw_color[:4], dtype=np.uint8),
                                (len(finding_mesh.faces), 1),
                            )
                            optimized_scene.add_geometry(finding_mesh, node_name=mesh_label, geom_name=mesh_label)
                            optimized_meshes.append((mesh_label, finding_mesh, rgba))
                            finding_meta_for_usdz[mesh_label] = {"id": finding.get("id", ""), "type": finding.get("type", "")}
                            findings_items.append({
                                "id": finding.get("id", ""),
                                "type": finding.get("type", ""),
                                "label": finding.get("label", ""),
                                "bone_label": finding.get("bone_label", ""),
                                "confidence": finding.get("confidence", 0.0),
                                "mesh_label": mesh_label,
                                "color": raw_color[:4],
                                "review_required": True,
                            })
                            break
                except Exception:
                    continue
        except (json.JSONDecodeError, KeyError):
            pass

    optimized_scene.export(glb_path, file_type="glb")
    _write_usdz(optimized_meshes, usdz_path, finding_metadata=finding_meta_for_usdz if finding_meta_for_usdz else None)

    labels_by_group: dict[str, list[str]] = {}
    for item in anatomy:
        labels_by_group.setdefault(item["group"], []).append(item["label"])

    finding_mesh_labels = [item["mesh_label"] for item in findings_items]
    presets = {
        "All": [item["label"] for item in anatomy] + finding_mesh_labels,
        "Chest Core": [
            label
            for group in ("lungs", "heart", "arteries", "veins", "airway")
            for label in labels_by_group.get(group, [])
        ],
        "Bones": labels_by_group.get("bones", []),
        "Vessels": labels_by_group.get("arteries", []) + labels_by_group.get("veins", []),
        "Lungs": labels_by_group.get("lungs", []),
        "Heart": labels_by_group.get("heart", []),
    }
    if finding_mesh_labels:
        presets["Findings"] = finding_mesh_labels
        presets["Bones + Findings"] = labels_by_group.get("bones", []) + finding_mesh_labels

    manifest = {
        "id": output_id,
        "kind": "visionos_asset",
        "source_volume_id": source_id,
        "quality": quality.name,
        "model_url": f"/generated/visionos/{usdz_path.name}",
        "usdz_url": f"/generated/visionos/{usdz_path.name}",
        "glb_url": f"/generated/visionos/{glb_path.name}",
        "manifest_url": f"/generated/visionos/{manifest_path.name}",
        "mesh_vertices": int(sum(item["vertices"] for item in anatomy)),
        "mesh_faces": int(sum(item["faces"] for item in anatomy)),
        "anatomy_count": len(anatomy),
        "anatomy": anatomy,
        "findings": findings_items,
        "presets": presets,
        "backend_routes": {
            "slices": "/api/slices",
            "analyze": "/api/analyze",
            "volumes": "/api/volumes",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_visionos_manifests() -> list[dict[str, Any]]:
    settings.ensure_generated_dirs()
    manifests: list[dict[str, Any]] = []
    for path in sorted(settings.GENERATED_VISIONOS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return manifests
