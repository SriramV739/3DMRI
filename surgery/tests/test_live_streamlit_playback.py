from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import interactive_vlm
from operators.sam2_bootstrap import VENDORED_MEDSAM2_ROOT


def _clear_modules(*prefixes: str) -> None:
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(module_name, None)


def _write_test_frame(path: Path, color_bgr: tuple[int, int, int]) -> None:
    frame = np.full((48, 64, 3), color_bgr, dtype=np.uint8)
    ok = cv2.imwrite(str(path), frame)
    assert ok


def _decode_mjpeg_frame(frame_bytes: bytes) -> np.ndarray:
    header_end = frame_bytes.find(b"\r\n\r\n")
    assert header_end != -1
    jpeg_bytes = frame_bytes[header_end + 4 :]
    if jpeg_bytes.endswith(b"\r\n"):
        jpeg_bytes = jpeg_bytes[:-2]
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


def _make_cfg(tmp_path: Path) -> dict:
    converted_root = tmp_path / "converted"
    frames_dir = converted_root / "clip_001" / "frames"
    frames_dir.mkdir(parents=True)
    _write_test_frame(frames_dir / "00000.jpg", (12, 34, 220))
    _write_test_frame(frames_dir / "00001.jpg", (16, 40, 200))
    return {
        "replayer": {
            "directory": str(converted_root),
            "basename": "clip_001/clip_001",
            "realtime": True,
            "frame_rate": 25.0,
        },
        "live_overlay": {
            "enabled": True,
            "update_every_n_frames": 1,
            "max_inference_fps": 5.0,
            "mask_stale_after_frames": 12,
            "mask_stale_after_seconds": 0.5,
            "replace_on_new_query": True,
        },
        "overlay": {
            "colors": {"gallbladder": [0, 255, 128, 160], "liver": [255, 215, 0, 160]},
            "blend_alpha": 0.45,
            "glow_effect": False,
            "glow_radius": 1,
            "contour_thickness": 1,
        },
    }


def test_segmenter_import_order_uses_vendored_sam2():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    _clear_modules("sam2", "operators.sam2_bootstrap", "operators.sam2_inference_op", "operators.medsam2_inference_op")

    vendored_path = str(VENDORED_MEDSAM2_ROOT.resolve())
    removed_vendored_path = vendored_path in sys.path
    if removed_vendored_path:
        sys.path.remove(vendored_path)

    site_sam2 = importlib.import_module("sam2")
    assert VENDORED_MEDSAM2_ROOT.resolve() not in Path(site_sam2.__file__).resolve().parents
    _clear_modules("sam2")
    if removed_vendored_path:
        sys.path.insert(0, vendored_path)

    importlib.import_module("operators.sam2_inference_op")
    importlib.import_module("operators.medsam2_inference_op")
    active_sam2 = importlib.import_module("sam2")

    assert VENDORED_MEDSAM2_ROOT.resolve() in Path(active_sam2.__file__).resolve().parents
    assert GlobalHydra.instance().is_initialized()


def test_run_mjpeg_loop_yields_video_frame_when_pipeline_bootstrap_degrades(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    state = interactive_vlm.StreamState("clip_001")

    monkeypatch.setattr(interactive_vlm, "build_detector", lambda cfg: (_ for _ in ()).throw(RuntimeError("detector down")))
    monkeypatch.setattr(interactive_vlm, "build_segmenter", lambda cfg: (_ for _ in ()).throw(RuntimeError("segmenter down")))
    monkeypatch.setattr(interactive_vlm, "build_scene_copilot", lambda cfg: None)

    interactive_vlm.start_pipeline_bootstrap(cfg, state)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        _, status, error = state.get_pipeline_snapshot()
        if status != "initializing":
            break
        time.sleep(0.01)

    assert status == "degraded"
    assert "detector unavailable" in error
    assert "segmenter unavailable" in error

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    frame = _decode_mjpeg_frame(next(generator))

    assert frame.mean() > 10
    assert frame[0, 0, 2] > frame[0, 0, 1]


def test_live_streaming_never_calls_prepare_video(tmp_path):
    cfg = _make_cfg(tmp_path)

    class FakeSegmenter:
        use_temporal_memory = True
        device = "cpu"

        def __init__(self):
            self.prepare_called = False

        def prepare_video(self, *_args, **_kwargs):
            self.prepare_called = True

        def segment_frame(self, *_args, **_kwargs):
            return {}

    fake_segmenter = FakeSegmenter()
    state = interactive_vlm.StreamState("clip_001")
    state.set_pipeline(
        interactive_vlm.LivePipeline(
            detector=None,
            vlm_guide=None,
            scene_copilot=None,
            segmenter=fake_segmenter,
            overlay=None,
            device="cpu",
        ),
        [],
    )

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    _decode_mjpeg_frame(next(generator))

    assert fake_segmenter.prepare_called is False


def test_live_streaming_does_not_run_inference_inline_without_overlay_query(tmp_path):
    cfg = _make_cfg(tmp_path)
    state = interactive_vlm.StreamState("clip_001")

    class FailingDetector:
        def detect(self, *_args, **_kwargs):
            raise AssertionError("continuous playback should not call detector inline")

    state.set_pipeline(
        interactive_vlm.LivePipeline(
            detector=FailingDetector(),
            vlm_guide=None,
            scene_copilot=None,
            segmenter=None,
            overlay=None,
            device="cpu",
        ),
        [],
    )

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    frame = _decode_mjpeg_frame(next(generator))

    assert frame.mean() > 10


def test_overlay_query_adds_persistent_masks_and_generation():
    state = interactive_vlm.StreamState("clip_001")
    state.start_overlay_query("highlight the liver", ["liver"])
    state.finish_overlay_job(
        generation=1,
        frame_idx=0,
        masks={"liver": torch.ones((8, 8), dtype=torch.float32)},
    )

    state.start_overlay_query("highlight the gallbladder", ["gallbladder"])

    active_query, labels, status, _, _, generation, completed_generation, _ = state.get_overlay_snapshot()
    masks, overlay_frame_idx, _ = state.get_latest_overlay_masks()

    assert active_query == "highlight the gallbladder"
    assert labels == ["liver", "gallbladder"]
    assert status == "running"
    assert generation == 2
    assert completed_generation == 1
    assert "liver" in masks
    assert overlay_frame_idx == -1


def test_overlay_key_removal_keeps_remaining_masks():
    state = interactive_vlm.StreamState("clip_001")
    state.start_overlay_query("highlight liver and gallbladder", ["liver", "gallbladder"])
    state.finish_overlay_job(
        generation=1,
        frame_idx=0,
        masks={
            "liver": torch.ones((8, 8), dtype=torch.float32),
            "gallbladder": torch.ones((8, 8), dtype=torch.float32),
        },
    )

    state.remove_overlay_labels(["liver"])

    _, labels, status, _, _, _, _, _ = state.get_overlay_snapshot()
    masks, _, _ = state.get_latest_overlay_masks()
    assert labels == ["gallbladder"]
    assert status == "running"
    assert "liver" not in masks


def test_temporary_overlay_expires(monkeypatch):
    state = interactive_vlm.StreamState("clip_001")
    now = [100.0]
    monkeypatch.setattr(interactive_vlm.time, "time", lambda: now[0])

    state.start_overlay_query("temporary focus", ["cystic_duct"], mode="temporary", temporary_seconds=2.0)
    _, labels, _, _, _, _, _, _ = state.get_overlay_snapshot()
    assert labels == ["cystic_duct"]

    now[0] = 103.0
    _, labels, status, _, _, _, _, _ = state.get_overlay_snapshot()
    assert labels == []
    assert status == "idle"


def test_clear_overlay_removes_active_query_and_masks():
    state = interactive_vlm.StreamState("clip_001")
    state.start_overlay_query("highlight the liver", ["liver"])
    state.finish_overlay_job(
        generation=1,
        frame_idx=0,
        masks={"liver": torch.ones((8, 8), dtype=torch.float32)},
    )

    state.clear_overlay()
    active_query, labels, status, _, _, _, _, _ = state.get_overlay_snapshot()
    masks, overlay_frame_idx, _ = state.get_latest_overlay_masks()

    assert active_query == ""
    assert labels == []
    assert status == "idle"
    assert masks == {}
    assert overlay_frame_idx == -1


def test_overlay_frame_queue_keeps_newest_pending_frame():
    state = interactive_vlm.StreamState("clip_001")
    state.start_overlay_query("highlight the liver", ["liver"])
    older = np.full((8, 8, 3), 10, dtype=np.uint8)
    newer = np.full((8, 8, 3), 90, dtype=np.uint8)

    state.offer_overlay_frame(0, older, update_every_n_frames=1)
    state.offer_overlay_frame(5, newer, update_every_n_frames=1)
    job = state.claim_overlay_job()

    assert job is not None
    frame_idx, source_bgr, generation, query, labels = job
    assert frame_idx == 5
    assert int(source_bgr[0, 0, 0]) == 90
    assert generation == 1
    assert query == "highlight the liver"
    assert labels == ["liver"]


def test_live_overlay_renders_fresh_masks_on_stream(tmp_path):
    cfg = _make_cfg(tmp_path)
    state = interactive_vlm.StreamState("clip_001")

    class FakeOverlay:
        def composite(self, frame_t, _masks):
            frame_np = frame_t.cpu().numpy().copy()
            frame_np[:, :] = [200, 50, 25]
            alpha = np.full(frame_np.shape[:2] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([frame_np.astype(np.uint8), alpha], axis=2)
            return torch.from_numpy(rgba)

    state.set_pipeline(
        interactive_vlm.LivePipeline(
            detector=None,
            vlm_guide=None,
            scene_copilot=None,
            segmenter=None,
            overlay=FakeOverlay(),
            device="cpu",
        ),
        [],
    )

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    _decode_mjpeg_frame(next(generator))

    state.start_overlay_query("highlight the liver", ["liver"])
    state.finish_overlay_job(
        generation=1,
        frame_idx=1,
        masks={"liver": torch.ones((48, 64), dtype=torch.float32)},
    )

    frame = _decode_mjpeg_frame(next(generator))
    assert frame[0, 0, 2] > 150


def test_live_overlay_ignores_stale_masks(tmp_path):
    cfg = _make_cfg(tmp_path)
    state = interactive_vlm.StreamState("clip_001")

    class FakeOverlay:
        def composite(self, *_args, **_kwargs):
            raise AssertionError("stale masks should not be composited")

    state.set_pipeline(
        interactive_vlm.LivePipeline(
            detector=None,
            vlm_guide=None,
            scene_copilot=None,
            segmenter=None,
            overlay=FakeOverlay(),
            device="cpu",
        ),
        [],
    )

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    _decode_mjpeg_frame(next(generator))

    state.start_overlay_query("highlight the liver", ["liver"])
    state.finish_overlay_job(
        generation=1,
        frame_idx=-100,
        masks={"liver": torch.ones((48, 64), dtype=torch.float32)},
    )

    frame = _decode_mjpeg_frame(next(generator))
    assert frame.mean() > 10


def test_build_overlay_fallback_detections_covers_liver_and_gallbladder():
    liver = interactive_vlm.build_overlay_fallback_detections(
        requested_labels=["liver"],
        frame_shape=(480, 854, 3),
    )
    gallbladder = interactive_vlm.build_overlay_fallback_detections(
        requested_labels=["gallbladder"],
        frame_shape=(480, 854, 3),
    )

    assert [det.class_name for det in liver] == ["liver", "liver"]
    assert liver[0].bbox[2] < 0.5 * 854
    assert liver[1].bbox[0] > 0.5 * 854
    assert [det.class_name for det in gallbladder] == ["gallbladder"]
    assert gallbladder[0].bbox[1] == 0.0


def test_build_detector_passes_roboflow_class_mapping(monkeypatch):
    captured = {}

    class FakeRoboflowDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(interactive_vlm, "RoboflowHostedDetector", FakeRoboflowDetector)

    cfg = {
        "detector": {"backend": "roboflow_hosted"},
        "roboflow_laparoscopy": {
            "model_id": "laparoscopy/14",
            "api_url": "https://serverless.roboflow.com",
            "api_key": "key",
            "api_key_env": "ROBOFLOW_API_KEY",
            "confidence_threshold": 0.35,
            "detect_every_n_frames": 15,
            "target_classes": ["gallbladder", "liver"],
            "class_name_map": {"Gallbladder": "gallbladder", "Liver": "liver"},
        },
    }

    interactive_vlm.build_detector(cfg)

    assert captured["target_classes"] == ["gallbladder", "liver"]
    assert captured["class_name_map"]["Gallbladder"] == "gallbladder"
    assert captured["detect_every_n_frames"] == 15


def test_live_streaming_paces_frames_using_replayer_frame_rate(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    state = interactive_vlm.StreamState("clip_001")
    sleep_calls: list[float] = []
    perf_values = iter([0.0, 0.0, 0.0, 0.0, 0.01])

    monkeypatch.setattr(interactive_vlm.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(interactive_vlm.time, "perf_counter", lambda: next(perf_values))

    generator = interactive_vlm.run_mjpeg_loop(cfg, state)
    next(generator)
    next(generator)
    _decode_mjpeg_frame(next(generator))
    _decode_mjpeg_frame(next(generator))

    assert any(seconds >= 0.03 for seconds in sleep_calls)
