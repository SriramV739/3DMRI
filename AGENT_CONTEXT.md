# Agent Handoff Context

Last updated: 2026-04-19

## Project Summary

This repo contains a Streamlit surgical AR/VLM demo under `surgery/`. The main live app is:

```powershell
cd surgery
python -m streamlit run scripts/interactive_vlm.py --server.headless true --server.port 8523
```

The app serves a Streamlit UI on port `8523` and an MJPEG video feed on port `8503`.

## Current Demo Target

The app is configured to replay a stitched Dr. R. K. Mishra laparoscopic cholecystectomy sequence built from clips `002` through `011`.

Configured in `surgery/config/app_config.yaml`:

```yaml
replayer:
  directory: "../data/converted"
  basename: "dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_clips_002_011_stitched/dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_clips_002_011_stitched"
```

The generated stitched frames and MP4 are intentionally ignored by Git because local generated data is large. To regenerate the stitched asset from existing converted clips:

```powershell
python surgery/scripts/stitch_mishra_clips.py --converted-root surgery/data/converted --start-clip 2 --end-clip 11
```

Expected stitched asset:

- Folder: `surgery/data/converted/dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_clips_002_011_stitched`
- Frames: `8453` JPEGs, `00000.jpg` through `08452.jpg`
- MP4: `dangerous_way_of_performing_laparoscopic_cholecystectomy_dr_r_k_mishra_1080p_clips_002_011_stitched.mp4`

## Required Local Assets

These are ignored and must exist locally for the full demo:

- `surgery/data/converted/...` converted frame folders.
- `surgery/models/MedSAM2_latest.pt`.
- `surgery/models/yolov8n.pt` if using the local YOLO backend.
- Groq and Roboflow keys in environment variables or `surgery/.env.local`.

Use `.env.example` as the public template:

```powershell
$env:GROQ_API_KEY="..."
$env:ROBOFLOW_API_KEY="..."
```

Do not commit real API keys.

## Major Implemented Changes

- Live segmentation overlays run alongside the MJPEG video stream instead of blocking playback.
- Overlay command parsing supports switching/removing labels and typo aliases like `galbladder`.
- Persistent overlays are additive and remain until removed.
- Temporary overlays can be triggered from VLM recommended attention targets for short explanations.
- Highlight key overlays the video and includes inline `on` links to turn active highlights off.
- VLM prompt-box localization was added so requested labels outside detector support can still produce approximate boxes for MedSAM2.
- Surgery session logging writes JSONL events and keyframes.
- End-surgery report generation uses VLM when available and retries text-only if Groq rejects multimodal report payloads.
- UI is a glassmorphism surgical workspace: padded camera feed, glass top bar, right copilot rail, and custom left/right chat bubbles.
- Chat renderer no longer uses `st.chat_message` because Streamlit internals caused clipped/overlapping bubbles.

## Key Files

- `surgery/scripts/interactive_vlm.py`: main live Streamlit app, MJPEG server, overlay state, chat UI, report button.
- `surgery/operators/vlm_prompt_op.py`: VLM label selection and VLM prompt-box localization.
- `surgery/operators/scene_copilot_op.py`: surgeon-facing VLM responses and temporary attention targets.
- `surgery/operators/medsam2_inference_op.py`: MedSAM2 segmentation wrapper.
- `surgery/operators/overlay_compositor_op.py`: mask coloring/composition.
- `surgery/session/surgery_log.py`: append-only surgery event log and keyframe saving.
- `surgery/session/surgery_report.py`: final draft report generation.
- `surgery/scripts/stitch_mishra_clips.py`: regenerates stitched Mishra clips `002-011`.
- `surgery/config/app_config.yaml`: active runtime config.

## Current Config Notes

- `detector.backend` is `roboflow_hosted`.
- `vlm.provider` and `scene_copilot.provider` are `openai_compatible`.
- API keys are blank in committed config and should come from:
  - `GROQ_API_KEY`
  - `ROBOFLOW_API_KEY`
- `live_overlay.replace_on_new_query` is `false` so highlights are additive.
- `live_overlay.short_term_seconds` is `6.0`.
- `medsam2.max_objects` is `10`.

## Validation Commands

Fast focused validation:

```powershell
python -m py_compile surgery\scripts\interactive_vlm.py surgery\operators\vlm_prompt_op.py surgery\operators\scene_copilot_op.py surgery\operators\medsam2_inference_op.py surgery\session\surgery_report.py
python -m pytest surgery\tests\test_interactive_vlm.py surgery\tests\test_live_streamlit_playback.py surgery\tests\test_scene_copilot_op.py surgery\tests\test_surgery_log_report.py surgery\tests\test_groq_vlm.py -q
```

Last known focused result before cleanup:

```text
50 passed, 2 skipped
```

After cleanup and chat/stitch additions, the smaller UI/overlay suite passed:

```text
23 passed
```

## Cleanup Policy

The repo is now prepared so source/config/tests/docs are visible to Git, while large or local artifacts are ignored:

- Runtime logs: ignored and cleaned from `surgery/logs/`.
- Python caches: ignored and cleaned.
- Root `data/` surgery reports: ignored.
- Generated video/dataset folders under `surgery/data/`: ignored.
- Model weights/checkpoints: ignored.
- Real local env files: ignored.

Before pushing, run `git status --short --ignored` and verify no secrets or generated media are staged.
