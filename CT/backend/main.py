from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import settings
from backend.imaging.dicom_io import discover_slices, ensure_slice_preview_png
from backend.imaging.totalseg_pipeline import generate_totalseg_chest
from backend.imaging.visionos_export import export_visionos_asset, load_visionos_manifests
from backend.imaging.bone_findings_pipeline import load_bone_findings_manifest, run_bone_findings
from backend.imaging.volume_pipeline import generate_batch, load_volume_manifests
from backend.vlm_client import analyze_slices, analyze_slices_gemini, analyze_snapshot_gemini, test_gemini_key


app = FastAPI(title="CT Spatial VLM Imaging API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.ensure_generated_dirs()
FRONTEND_DIST_DIR = settings.BASE_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
app.mount("/generated", StaticFiles(directory=settings.GENERATED_DIR), name="generated")
if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="frontend-assets")


class AnalyzeRequest(BaseModel):
    slice_ids: list[str]
    user_note: str | None = None
    dry_run: bool = False
    provider: str = "groq"


class SnapshotAnalyzeRequest(BaseModel):
    image_base64: str
    user_note: str | None = None
    asset_id: str | None = None
    visible_labels: list[str] = []
    rotation_x: float | None = None
    rotation_y: float | None = None
    scale: float | None = None
    dry_run: bool = False


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "data_dir_exists": settings.DATA_DIR.exists(),
        "groq_api_key_loaded": bool(settings.GROQ_API_KEY),
        "gemini_api_key_loaded": bool(settings.GEMINI_API_KEY),
    }


@app.get("/api/slices")
def slices(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, object]]:
    result = []
    for record in discover_slices(limit=limit):
        result.append(
            {
                "id": record.slice_id,
                "age": record.age,
                "contrast": record.contrast,
                "dicom_name": record.dicom_name,
                "tiff_name": record.tiff_name,
                "series_uid": record.series_uid,
                "study_uid": record.study_uid,
                "instance_number": record.instance_number,
                "slice_location": record.slice_location,
                "thumbnail_url": f"/api/slices/{record.slice_id}/image",
                "preview_url": f"/api/slices/{record.slice_id}/image",
                "colored_url": f"/generated/slices/{record.slice_id}.png",
            }
        )
    return result


@app.get("/api/slices/{slice_id}/image")
def slice_image(slice_id: str) -> FileResponse:
    try:
        return FileResponse(ensure_slice_preview_png(slice_id), media_type="image/png")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Slice image not found") from exc


@app.get("/api/volumes")
def volumes() -> list[dict[str, object]]:
    return load_volume_manifests()


@app.get("/api/visionos/assets")
def visionos_assets() -> list[dict[str, object]]:
    return load_visionos_manifests()


@app.post("/api/process")
def process(limit: int = Query(default=10, ge=1, le=50), max_size: int = Query(default=176, ge=64, le=256)) -> list[dict[str, object]]:
    try:
        return generate_batch(limit=limit, max_size=max_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/process-totalseg")
def process_totalseg(
    run_segmentation: bool = Query(default=True),
    force_segmentation: bool = Query(default=False),
    device: str = Query(default="mps", pattern="^(mps|cpu|gpu)$"),
    fast: bool = Query(default=True),
) -> dict[str, object]:
    try:
        return generate_totalseg_chest(
            run_segmentation=run_segmentation,
            force_segmentation=force_segmentation,
            device=device,
            fast=fast,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/visionos/export")
def process_visionos_export(
    source_id: str = Query(default="totalseg_CT_chest_realistic"),
    quality: str = Query(default="balanced", pattern="^(preview|balanced|hq)$"),
    force: bool = Query(default=False),
) -> dict[str, object]:
    try:
        return export_visionos_asset(source_id=source_id, quality_name=quality, force=force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze_json(request: AnalyzeRequest) -> dict[str, object]:
    if not request.slice_ids:
        raise HTTPException(status_code=422, detail="slice_ids must contain at least one slice id")
    try:
        if request.provider == "gemini":
            return analyze_slices_gemini(request.slice_ids, user_note=request.user_note, dry_run=request.dry_run)
        return analyze_slices(request.slice_ids, user_note=request.user_note, dry_run=request.dry_run)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{request.provider} analysis failed: {type(exc).__name__}: {exc}") from exc


@app.post("/api/analyze-3d-snapshot")
def analyze_3d_snapshot(request: SnapshotAnalyzeRequest) -> dict[str, object]:
    if not request.image_base64.strip():
        raise HTTPException(status_code=422, detail="image_base64 must contain an image snapshot")
    try:
        return analyze_snapshot_gemini(
            image_base64=request.image_base64,
            user_note=request.user_note,
            asset_id=request.asset_id,
            visible_labels=request.visible_labels,
            rotation_x=request.rotation_x,
            rotation_y=request.rotation_y,
            scale=request.scale,
            dry_run=request.dry_run,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"gemini 3D snapshot analysis failed: {type(exc).__name__}: {exc}") from exc


@app.post("/api/findings/bones/run")
def findings_bones_run(
    source_id: str = Query(default="totalseg_CT_chest_realistic"),
    force: bool = Query(default=False),
    bone_labels: list[str] | None = Query(default=None),
    min_confidence: float = Query(default=0.55, ge=0.0, le=1.0),
) -> dict[str, object]:
    try:
        return run_bone_findings(
            source_id=source_id,
            force=force,
            bone_labels=bone_labels or None,
            min_confidence=min_confidence,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bone findings failed: {type(exc).__name__}: {exc}") from exc


@app.get("/api/findings/bones/{source_id}")
def findings_bones_get(source_id: str) -> dict[str, object]:
    try:
        return load_bone_findings_manifest(source_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/gemini-test")
def gemini_test() -> dict[str, object]:
    try:
        return test_gemini_key()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze-form")
def analyze_form(
    slice_ids: Annotated[list[str], Form()],
    user_note: Annotated[str | None, Form()] = None,
    dry_run: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    if not slice_ids:
        raise HTTPException(status_code=422, detail="slice_ids must contain at least one slice id")
    return analyze_slices(slice_ids, user_note=user_note, dry_run=dry_run)


@app.get("/", response_model=None)
def root():
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"message": "CT Spatial VLM Imaging API. See /docs for API routes."}
