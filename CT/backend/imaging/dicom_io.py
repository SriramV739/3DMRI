from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image

from backend import settings


@dataclass(frozen=True)
class SliceRecord:
    slice_id: str
    age: int | None
    contrast: bool | None
    dicom_name: str
    tiff_name: str | None
    dicom_path: Path
    tiff_path: Path | None
    series_uid: str | None
    study_uid: str | None
    instance_number: float | None
    slice_location: float | None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def load_overview() -> dict[str, dict[str, str]]:
    if not settings.OVERVIEW_CSV.exists():
        return {}
    with settings.OVERVIEW_CSV.open("r", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row.get("dicom_name", ""): row for row in rows if row.get("dicom_name")}


def discover_slices(limit: int | None = None) -> list[SliceRecord]:
    overview = load_overview()
    records: list[SliceRecord] = []
    for dicom_path in sorted(settings.DICOM_DIR.glob("*.dcm")):
        row = overview.get(dicom_path.name, {})
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        tiff_name = row.get("tiff_name") or None
        tiff_path = settings.TIFF_DIR / tiff_name if tiff_name else None
        records.append(
            SliceRecord(
                slice_id=dicom_path.stem,
                age=_to_int(row.get("Age")),
                contrast=_to_bool(row.get("Contrast")),
                dicom_name=dicom_path.name,
                tiff_name=tiff_name,
                dicom_path=dicom_path,
                tiff_path=tiff_path if tiff_path and tiff_path.exists() else None,
                series_uid=str(getattr(ds, "SeriesInstanceUID", "")) or None,
                study_uid=str(getattr(ds, "StudyInstanceUID", "")) or None,
                instance_number=_to_float(getattr(ds, "InstanceNumber", None)),
                slice_location=_to_float(getattr(ds, "SliceLocation", None)),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def get_slice_record(slice_id: str) -> SliceRecord:
    for record in discover_slices():
        if record.slice_id == slice_id:
            return record
    raise FileNotFoundError(f"Unknown CT slice id: {slice_id}")


def read_hu_slice(dicom_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    ds = pydicom.dcmread(dicom_path, force=True)
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixels * slope + intercept
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    spacing = (
        float(getattr(ds, "SliceThickness", 1.0) or 1.0),
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
    )
    meta = {
        "rows": int(getattr(ds, "Rows", hu.shape[0])),
        "columns": int(getattr(ds, "Columns", hu.shape[1])),
        "spacing": spacing,
        "slope": slope,
        "intercept": intercept,
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "instance_number": _to_float(getattr(ds, "InstanceNumber", None)),
        "slice_location": _to_float(getattr(ds, "SliceLocation", None)),
    }
    return hu, meta


def window_to_uint8(hu: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(hu, low, high)
    return ((clipped - low) / (high - low) * 255).astype(np.uint8)


def pseudo_color_hu(hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(hu, -1000, 1200)
    normalized = (clipped + 1000) / 2200

    rgb = np.zeros((*hu.shape, 3), dtype=np.float32)
    air = clipped < -800
    lung = (clipped >= -800) & (clipped < -250)
    soft = (clipped >= -250) & (clipped < 180)
    enhanced = (clipped >= 180) & (clipped < 500)
    bone = clipped >= 500

    rgb[air] = [5, 11, 18]
    rgb[lung] = [57, 125, 176]
    rgb[soft] = [211, 122, 116]
    rgb[enhanced] = [234, 185, 96]
    rgb[bone] = [245, 241, 214]

    shade = (0.45 + normalized[..., None] * 0.55).astype(np.float32)
    return np.clip(rgb * shade, 0, 255).astype(np.uint8)


def save_colored_slice_png(hu: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pseudo_color_hu(hu), mode="RGB").save(destination)


def save_windowed_slice_png(hu: np.ndarray, destination: Path, center: float = 40.0, width: float = 400.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(window_to_uint8(hu, center=center, width=width), mode="L").save(destination)


def ensure_slice_preview_png(slice_id: str) -> Path:
    destination = settings.GENERATED_SLICES_DIR / f"{slice_id}_preview.png"
    if destination.exists():
        return destination
    record = get_slice_record(slice_id)
    hu, _meta = read_hu_slice(record.dicom_path)
    save_windowed_slice_png(hu, destination)
    return destination
