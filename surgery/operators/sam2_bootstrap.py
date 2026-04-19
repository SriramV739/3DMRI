"""
Helpers for loading the vendored SAM2/MedSAM2 package consistently.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_MEDSAM2_ROOT = REPO_ROOT / "third_party" / "MedSAM2"
VENDORED_SAM2_ROOT = VENDORED_MEDSAM2_ROOT / "sam2"
VENDORED_SAM2_CONFIG_ROOT = VENDORED_SAM2_ROOT / "configs"


def _purge_sam2_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "sam2" or module_name.startswith("sam2."):
            sys.modules.pop(module_name, None)


def _ensure_vendored_path_first() -> None:
    if not VENDORED_MEDSAM2_ROOT.exists():
        raise ImportError(f"Vendored MedSAM2 package not found at {VENDORED_MEDSAM2_ROOT}")

    vendored_path = str(VENDORED_MEDSAM2_ROOT)
    if vendored_path in sys.path:
        sys.path.remove(vendored_path)
    sys.path.insert(0, vendored_path)


def ensure_vendored_sam2(*, reset_hydra: bool = True):
    """Import the vendored `sam2` package and bind Hydra to it."""
    _ensure_vendored_path_first()

    loaded = sys.modules.get("sam2")
    loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
    if loaded_file and VENDORED_MEDSAM2_ROOT.resolve() not in loaded_file.parents:
        _purge_sam2_modules()

    if reset_hydra and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    sam2_pkg = importlib.import_module("sam2")
    sam2_file = Path(getattr(sam2_pkg, "__file__", "")).resolve()
    if VENDORED_MEDSAM2_ROOT.resolve() not in sam2_file.parents:
        raise ImportError(f"Expected vendored sam2 package, found {sam2_file}")

    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("sam2", version_base="1.2")

    return sam2_pkg


def import_sam2_symbols(*, include_video_predictor: bool = False):
    """Load SAM2 symbols from the vendored package."""
    ensure_vendored_sam2(reset_hydra=False)
    build_module = importlib.import_module("sam2.build_sam")
    predictor_module = importlib.import_module("sam2.sam2_image_predictor")

    build_sam2 = build_module.build_sam2
    video_predictor = getattr(build_module, "build_sam2_video_predictor", None)
    image_predictor = predictor_module.SAM2ImagePredictor

    if include_video_predictor:
        if video_predictor is None:
            raise ImportError("Vendored sam2.build_sam is missing build_sam2_video_predictor")
        return build_sam2, video_predictor, image_predictor
    return build_sam2, image_predictor


def resolve_repo_path(raw_path: str, *, extra_roots: Iterable[Path] = ()) -> str:
    """Resolve checkpoints and other filesystem paths from repo-relative inputs."""
    candidate = Path(raw_path)
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                REPO_ROOT / raw_path,
                VENDORED_MEDSAM2_ROOT / raw_path,
                *[root / raw_path for root in extra_roots],
            ]
        )

    for path in search_paths:
        if path.exists():
            return str(path.resolve())

    model_fallback = REPO_ROOT / "models" / candidate.name
    if model_fallback.exists():
        return str(model_fallback.resolve())

    return raw_path


def normalize_sam_config_name(raw_path: str) -> str:
    """
    Convert filesystem-ish config inputs into the Hydra config name expected by vendored sam2.
    """
    candidate = Path(raw_path)
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                REPO_ROOT / raw_path,
                VENDORED_MEDSAM2_ROOT / raw_path,
                VENDORED_SAM2_ROOT / raw_path,
                VENDORED_SAM2_CONFIG_ROOT / candidate.name,
            ]
        )

    for path in search_paths:
        if path.exists():
            resolved = path.resolve()
            if VENDORED_SAM2_CONFIG_ROOT.resolve() in resolved.parents:
                rel = resolved.relative_to(VENDORED_SAM2_ROOT.resolve())
                return rel.as_posix()
            return raw_path.replace("\\", "/")

    if raw_path.startswith("configs/") or raw_path.startswith("configs\\"):
        return raw_path.replace("\\", "/")

    named_config = VENDORED_SAM2_CONFIG_ROOT / candidate.name
    if named_config.exists():
        return f"configs/{candidate.name}"

    return raw_path.replace("\\", "/")
