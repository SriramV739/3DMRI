from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters, measure
import trimesh


OUTPUT_DIR = Path("generated/test_renders/synthetic_rib_fracture")
VOLUME_SHAPE = (112, 160, 160)  # z, y, x
BONE_LABEL = 1
HIGHLIGHT_LABEL = 1


def _coord_to_index(x: float, y: float, z: float) -> tuple[int, int, int]:
    nz, ny, nx = VOLUME_SHAPE
    ix = int(round((x + 1.0) * 0.5 * (nx - 1)))
    iy = int(round((y + 1.0) * 0.5 * (ny - 1)))
    iz = int(round((z + 1.0) * 0.5 * (nz - 1)))
    return iz, iy, ix


def _draw_sphere(volume: np.ndarray, center: tuple[float, float, float], radius: float, label: int) -> None:
    x, y, z = center
    iz, iy, ix = _coord_to_index(x, y, z)
    nz, ny, nx = volume.shape
    rx = max(2, int(math.ceil(radius * (nx - 1) / 2.0)) + 1)
    ry = max(2, int(math.ceil(radius * (ny - 1) / 2.0)) + 1)
    rz = max(2, int(math.ceil(radius * (nz - 1) / 2.0)) + 1)

    z0, z1 = max(0, iz - rz), min(nz, iz + rz + 1)
    y0, y1 = max(0, iy - ry), min(ny, iy + ry + 1)
    x0, x1 = max(0, ix - rx), min(nx, ix + rx + 1)

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    dx = (xx - ix) * 2.0 / (nx - 1)
    dy = (yy - iy) * 2.0 / (ny - 1)
    dz = (zz - iz) * 2.0 / (nz - 1)
    mask = (dx * dx + dy * dy + dz * dz) <= radius * radius
    region = volume[z0:z1, y0:y1, x0:x1]
    region[mask] = label


def _rib_center(side: int, rib_index: int, t: float, displaced: bool = False) -> tuple[float, float, float]:
    x = side * (0.32 + 0.42 * math.cos(t))
    y = -0.08 + 0.54 * math.sin(t)
    z = -0.56 + rib_index * 0.125 + 0.018 * math.sin(t * 2.0)
    if displaced:
        x += side * 0.085
        y += 0.165
        z -= 0.075
    return x, y, z


def _draw_jagged_fracture_end(
    volume: np.ndarray,
    side: int,
    rib_index: int,
    t: float,
    displaced: bool,
    direction: int,
) -> None:
    base = np.array(_rib_center(side, rib_index, t, displaced=displaced))
    offsets = [
        (0.000, 0.000, 0.000, 0.035),
        (0.020 * side, 0.020, -0.014, 0.020),
        (-0.016 * side, -0.012, 0.018, 0.018),
        (0.030 * side, -0.004, 0.010, 0.015),
        (0.018 * side * direction, 0.030 * direction, -0.020, 0.014),
        (-0.010 * side * direction, 0.018, 0.025 * direction, 0.013),
    ]
    for dx, dy, dz, radius in offsets:
        _draw_sphere(volume, tuple(base + np.array([dx, dy, dz])), radius, BONE_LABEL)


def _generate_synthetic_volume() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bone_volume = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    highlight_volume = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    broken_side = -1
    broken_rib = 5
    gap_min = -0.10
    gap_max = 0.28
    tube_radius = 0.028
    fracture_end_radius = 0.042

    for rib_index in range(9):
        for side in (-1, 1):
            for t in np.linspace(-1.12, 1.16, 136):
                is_broken = side == broken_side and rib_index == broken_rib
                if is_broken and gap_min < t < gap_max:
                    continue
                displaced = is_broken and t >= gap_max
                _draw_sphere(bone_volume, _rib_center(side, rib_index, float(t), displaced), tube_radius, BONE_LABEL)

    for z in np.linspace(-0.68, 0.64, 64):
        _draw_sphere(bone_volume, (0.0, -0.54, float(z)), 0.036, BONE_LABEL)
        _draw_sphere(bone_volume, (0.0, 0.46, float(z)), 0.045, BONE_LABEL)

    _draw_jagged_fracture_end(bone_volume, broken_side, broken_rib, gap_min, displaced=False, direction=-1)
    _draw_jagged_fracture_end(bone_volume, broken_side, broken_rib, gap_max, displaced=True, direction=1)

    left_end = _rib_center(broken_side, broken_rib, gap_min, displaced=False)
    right_end = _rib_center(broken_side, broken_rib, gap_max, displaced=True)
    _draw_sphere(highlight_volume, left_end, fracture_end_radius, HIGHLIGHT_LABEL)
    _draw_sphere(highlight_volume, right_end, fracture_end_radius, HIGHLIGHT_LABEL)

    gap_center = _rib_center(broken_side, broken_rib, (gap_min + gap_max) / 2.0, displaced=False)
    metadata = {
        "phantom": "synthetic_rib_fracture_ct",
        "abnormality": "left displaced jagged rib fracture",
        "broken_side": "left",
        "broken_rib_index": broken_rib,
        "bone_only_render": True,
        "annotation_is_optional": True,
        "gap_center_xyz": [round(v, 4) for v in gap_center],
        "fracture_endpoints_xyz": [[round(v, 4) for v in left_end], [round(v, 4) for v in right_end]],
        "labels": {"bone": "rib cage mesh with actual fracture gap", "fracture_highlight": "optional red overlay"},
    }
    return bone_volume, highlight_volume, metadata


def _mesh_for_label(volume: np.ndarray, label: int, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mask = (volume == label).astype(np.float32)
    smoothed = filters.gaussian(mask, sigma=0.85, preserve_range=True)
    verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.18, spacing=(1.0, 1.0, 1.0))
    nz, ny, nx = volume.shape
    xyz = np.column_stack(
        (
            (verts[:, 2] / (nx - 1) - 0.5) * 2.0,
            (verts[:, 1] / (ny - 1) - 0.5) * 2.0,
            (verts[:, 0] / (nz - 1) - 0.5) * 2.0,
        )
    )
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=True)
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=8)
    mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _save_projection_preview(bone_volume: np.ndarray, highlight_volume: np.ndarray, output_path: Path) -> int:
    projection = np.zeros((VOLUME_SHAPE[0], VOLUME_SHAPE[2], 3), dtype=np.uint8)
    bone = np.max(bone_volume == BONE_LABEL, axis=1)
    fracture = np.max(highlight_volume == HIGHLIGHT_LABEL, axis=1)
    projection[bone] = (222, 218, 204)
    projection[fracture] = (235, 64, 52)
    projection = np.flipud(projection)
    image = Image.fromarray(projection, mode="RGB").resize((960, 672), Image.Resampling.BICUBIC)
    image.save(output_path)
    return int(np.count_nonzero(fracture))


def _rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    yaw_matrix = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    pitch_matrix = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return pitch_matrix @ yaw_matrix


def _save_mesh_preview(
    bone_mesh: trimesh.Trimesh,
    output_path: Path,
    fracture_mesh: trimesh.Trimesh | None = None,
    zoom_center: tuple[float, float, float] = (-0.58, 0.06, 0.08),
    scale: float = 830.0,
) -> int:
    width, height = 1200, 820
    image = Image.new("RGB", (width, height), (8, 10, 14))
    pixels = image.load()
    rotation = _rotation_matrix(yaw=math.radians(-47), pitch=math.radians(-11))
    center = np.array(zoom_center)

    def project(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotated = (vertices - center) @ rotation.T
        px = np.round(width * 0.5 + rotated[:, 0] * scale).astype(np.int32)
        py = np.round(height * 0.54 - rotated[:, 2] * scale).astype(np.int32)
        depth = rotated[:, 1]
        return px, py, depth

    bone_px, bone_py, bone_depth = project(np.asarray(bone_mesh.vertices))
    order = np.argsort(bone_depth)
    low, high = float(np.min(bone_depth)), float(np.max(bone_depth))
    span = max(0.001, high - low)
    for idx in order:
        x, y = int(bone_px[idx]), int(bone_py[idx])
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            shade = int(174 + 58 * ((bone_depth[idx] - low) / span))
            color = (shade, shade - 4, shade - 18)
            pixels[x, y] = color
            pixels[x + 1, y] = color
            pixels[x, y + 1] = color

    red_pixels = 0
    if fracture_mesh is not None:
        fracture_px, fracture_py, fracture_depth = project(np.asarray(fracture_mesh.vertices))
        for idx in np.argsort(fracture_depth):
            x, y = int(fracture_px[idx]), int(fracture_py[idx])
            for yy in range(y - 4, y + 5):
                for xx in range(x - 4, x + 5):
                    if 0 <= xx < width and 0 <= yy < height and (xx - x) ** 2 + (yy - y) ** 2 <= 16:
                        pixels[xx, yy] = (240, 58, 48)
                        red_pixels += 1

    image.save(output_path)
    return red_pixels


def _save_fracture_closeup_preview(
    bone_mesh: trimesh.Trimesh,
    output_path: Path,
    fracture_mesh: trimesh.Trimesh | None = None,
) -> int:
    vertices = np.asarray(bone_mesh.vertices)
    fracture_center = np.array([-0.79, 0.03, 0.05])
    window = (
        (vertices[:, 0] > -0.96)
        & (vertices[:, 0] < -0.53)
        & (vertices[:, 1] > -0.26)
        & (vertices[:, 1] < 0.35)
        & (vertices[:, 2] > -0.05)
        & (vertices[:, 2] < 0.20)
    )
    close_vertices = vertices[window]

    width, height = 1200, 820
    image = Image.new("RGB", (width, height), (8, 10, 14))
    pixels = image.load()
    rotation = _rotation_matrix(yaw=math.radians(-82), pitch=math.radians(-5))
    scale = 1320.0

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotated = (points - fracture_center) @ rotation.T
        px = np.round(width * 0.52 + rotated[:, 0] * scale).astype(np.int32)
        py = np.round(height * 0.54 - rotated[:, 2] * scale).astype(np.int32)
        return px, py, rotated[:, 1]

    if len(close_vertices):
        px, py, depth = project(close_vertices)
        low, high = float(np.min(depth)), float(np.max(depth))
        span = max(0.001, high - low)
        for idx in np.argsort(depth):
            x, y = int(px[idx]), int(py[idx])
            if 2 <= x < width - 2 and 2 <= y < height - 2:
                shade = int(158 + 74 * ((depth[idx] - low) / span))
                color = (shade, shade - 4, shade - 17)
                for yy in range(y - 1, y + 2):
                    for xx in range(x - 1, x + 2):
                        pixels[xx, yy] = color

    red_pixels = 0
    if fracture_mesh is not None:
        highlight_vertices = np.asarray(fracture_mesh.vertices)
        px, py, depth = project(highlight_vertices)
        for idx in np.argsort(depth):
            x, y = int(px[idx]), int(py[idx])
            for yy in range(y - 5, y + 6):
                for xx in range(x - 5, x + 6):
                    if 0 <= xx < width and 0 <= yy < height and (xx - x) ** 2 + (yy - y) ** 2 <= 25:
                        pixels[xx, yy] = (240, 58, 48)
                        red_pixels += 1

    image.save(output_path)
    return red_pixels


def generate_standalone_rib_fracture_render() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bone_volume, highlight_volume, metadata = _generate_synthetic_volume()

    bone_mesh = _mesh_for_label(bone_volume, BONE_LABEL, (224, 220, 206, 230))
    fracture_mesh = _mesh_for_label(highlight_volume, HIGHLIGHT_LABEL, (235, 64, 52, 255))

    bone_only_scene = trimesh.Scene()
    bone_only_scene.add_geometry(bone_mesh.copy(), node_name="rib_cage_with_visible_jagged_fracture")

    annotated_scene = trimesh.Scene()
    annotated_scene.add_geometry(bone_mesh.copy(), node_name="rib_cage_with_visible_jagged_fracture")
    annotated_scene.add_geometry(fracture_mesh, node_name="optional_fracture_highlight_red")

    bone_only_glb_path = OUTPUT_DIR / "realistic_left_rib_fracture_bone_only.glb"
    annotated_glb_path = OUTPUT_DIR / "realistic_left_rib_fracture_annotated.glb"
    bone_only_preview_path = OUTPUT_DIR / "realistic_left_rib_fracture_bone_only_preview.png"
    annotated_preview_path = OUTPUT_DIR / "realistic_left_rib_fracture_annotated_preview.png"
    bone_only_closeup_path = OUTPUT_DIR / "realistic_left_rib_fracture_bone_only_closeup.png"
    annotated_closeup_path = OUTPUT_DIR / "realistic_left_rib_fracture_annotated_closeup.png"
    projection_path = OUTPUT_DIR / "synthetic_left_rib_fracture_projection.png"
    metadata_path = OUTPUT_DIR / "synthetic_left_rib_fracture_metadata.json"
    bone_only_scene.export(bone_only_glb_path)
    annotated_scene.export(annotated_glb_path)

    fracture_projection_pixels = _save_projection_preview(bone_volume, highlight_volume, projection_path)
    bone_only_red_pixels = _save_mesh_preview(bone_mesh, bone_only_preview_path, fracture_mesh=None)
    fracture_render_pixels = _save_mesh_preview(bone_mesh, annotated_preview_path, fracture_mesh=fracture_mesh)
    bone_only_closeup_red_pixels = _save_fracture_closeup_preview(
        bone_mesh,
        bone_only_closeup_path,
        fracture_mesh=None,
    )
    annotated_closeup_red_pixels = _save_fracture_closeup_preview(
        bone_mesh,
        annotated_closeup_path,
        fracture_mesh=fracture_mesh,
    )
    gap_iz, gap_iy, gap_ix = _coord_to_index(*metadata["gap_center_xyz"])
    gap_patch = bone_volume[
        max(0, gap_iz - 2) : gap_iz + 3,
        max(0, gap_iy - 2) : gap_iy + 3,
        max(0, gap_ix - 2) : gap_ix + 3,
    ]
    metrics = {
        "bone_vertices": int(len(bone_mesh.vertices)),
        "fracture_vertices": int(len(fracture_mesh.vertices)),
        "fracture_projection_pixels": fracture_projection_pixels,
        "bone_only_preview_red_pixels": bone_only_red_pixels,
        "bone_only_closeup_red_pixels": bone_only_closeup_red_pixels,
        "fracture_render_pixels": fracture_render_pixels,
        "annotated_closeup_red_pixels": annotated_closeup_red_pixels,
        "gap_patch_bone_voxels": int(np.count_nonzero(gap_patch == BONE_LABEL)),
        "bone_only_glb_size_bytes": bone_only_glb_path.stat().st_size,
        "annotated_glb_size_bytes": annotated_glb_path.stat().st_size,
    }
    metadata["metrics"] = metrics
    metadata["artifacts"] = {
        "bone_only_glb": str(bone_only_glb_path),
        "annotated_glb": str(annotated_glb_path),
        "bone_only_preview_png": str(bone_only_preview_path),
        "annotated_preview_png": str(annotated_preview_path),
        "bone_only_closeup_png": str(bone_only_closeup_path),
        "annotated_closeup_png": str(annotated_closeup_path),
        "projection_png": str(projection_path),
        "metadata_json": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class SyntheticRibFractureRenderTest(unittest.TestCase):
    def test_generates_detectable_broken_rib_render(self) -> None:
        metadata = generate_standalone_rib_fracture_render()
        artifacts = metadata["artifacts"]
        metrics = metadata["metrics"]

        self.assertTrue(Path(artifacts["bone_only_glb"]).exists())
        self.assertTrue(Path(artifacts["annotated_glb"]).exists())
        self.assertTrue(Path(artifacts["bone_only_preview_png"]).exists())
        self.assertTrue(Path(artifacts["annotated_preview_png"]).exists())
        self.assertTrue(Path(artifacts["bone_only_closeup_png"]).exists())
        self.assertTrue(Path(artifacts["annotated_closeup_png"]).exists())
        self.assertTrue(Path(artifacts["projection_png"]).exists())
        self.assertTrue(Path(artifacts["metadata_json"]).exists())
        self.assertGreater(metrics["bone_only_glb_size_bytes"], 50_000)
        self.assertGreater(metrics["annotated_glb_size_bytes"], metrics["bone_only_glb_size_bytes"])
        self.assertGreater(metrics["bone_vertices"], 5_000)
        self.assertGreater(metrics["fracture_vertices"], 200)
        self.assertGreater(metrics["fracture_projection_pixels"], 40)
        self.assertEqual(metrics["bone_only_preview_red_pixels"], 0)
        self.assertEqual(metrics["bone_only_closeup_red_pixels"], 0)
        self.assertGreater(metrics["fracture_render_pixels"], 100)
        self.assertGreater(metrics["annotated_closeup_red_pixels"], 100)
        self.assertEqual(metrics["gap_patch_bone_voxels"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
