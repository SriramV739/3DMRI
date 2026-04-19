from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "CTImages"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_CHEST_NRRD = RAW_DATA_DIR / "CT-chest.nrrd"
DICOM_DIR = DATA_DIR / "dicom_dir"
TIFF_DIR = DATA_DIR / "tiff_images"
OVERVIEW_CSV = DATA_DIR / "overview.csv"
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_SLICES_DIR = GENERATED_DIR / "slices"
GENERATED_VOLUMES_DIR = GENERATED_DIR / "volumes"
GENERATED_TOTALSEG_DIR = GENERATED_DIR / "totalseg"
GENERATED_TOTALSEG_INPUTS_DIR = GENERATED_TOTALSEG_DIR / "converted_inputs"
GENERATED_TOTALSEG_SEGMENTATIONS_DIR = GENERATED_TOTALSEG_DIR / "segmentations"
GENERATED_TOTALSEG_MESHES_DIR = GENERATED_TOTALSEG_DIR / "meshes"
GENERATED_FINDINGS_DIR = GENERATED_DIR / "findings"
GENERATED_VISIONOS_DIR = GENERATED_DIR / "visionos"
VLM_CONFIG_PATH = BASE_DIR / "vlm_config.json"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def ensure_generated_dirs() -> None:
    GENERATED_SLICES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TOTALSEG_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TOTALSEG_SEGMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TOTALSEG_MESHES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_VISIONOS_DIR.mkdir(parents=True, exist_ok=True)
