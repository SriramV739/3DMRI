# CT Spatial VLM Viewer - Project Status

Last updated: 2026-04-18

## 1) Project Purpose
Build an end-to-end CT imaging application that:
- Converts CT data into interactive 3D renders.
- Supports anatomy-aware segmentation and colored organ meshes.
- Allows multimodal AI analysis from selected 2D slices plus user prompts.
- Works in a spatial/web context (Three.js + WebXR support).

## 2) Primary Goals
- Read CT data from the CT dataset and produce 3D outputs.
- Provide two rendering paths:
  - HU-based fast rendering for quick volume inspection.
  - TotalSegmentator-based anatomy rendering for organ-level control.
- Let users select slices, send them to a VLM endpoint, and view analysis output.
- Keep model and prompt behavior configurable through config and .env.
- Provide strong interaction controls for showing/hiding organs.

## 3) Current Architecture
### Backend (FastAPI)
- Serves API routes for:
  - health/status
  - slice listing + slice images
  - HU volume generation
  - TotalSegmentator processing
  - volume manifest listing
  - multimodal analysis (Gemini or Groq)
- Serves generated assets under /generated.

### Frontend (React + Three.js + XR)
- Split-pane interface:
  - 3D spatial volume viewer
  - 2D slice gallery + AI analysis panel
- Volume selection, processing controls, and analysis controls.
- WebXR AR/VR entry actions are present in the scene UI.

### AI Integration
- Provider selectable in UI (Gemini or Groq).
- Analysis endpoint accepts selected slice IDs + user prompt text.
- Uses config-driven prompts and model names from vlm_config.json.
- Uses API keys from .env.

## 4) What Is Implemented So Far
- CT slice discovery and preview serving are working.
- Batch HU render generation is working.
- TotalSegmentator chest pipeline is working and exports anatomy-colored GLB meshes.
- Gemini analysis path is wired and validated through live backend execution.
- Enter-to-submit in the analysis textarea is fixed.
- Analysis panel layout no longer overlaps image tiles.
- Organ visibility controls are implemented and enhanced:
  - Show all / Hide all / Invert / Focus chest core
  - Group-level show/hide
  - Group-level Only isolate mode
  - Per-organ toggle chips
  - Organ search/filter
  - Dynamic organ list based on anatomy actually present in the active segmentation

## 5) Current Behavior Notes
- Full organ controls appear when viewing a TotalSegmentator volume (kind=totalsegmentator).
- If an anatomy is absent in the segmentation result, it is not shown as an available control.
- Non-TotalSeg volumes are HU isosurfaces, so organ-level toggles are not enabled there.

## 6) Runtime Model Configuration
- Current Groq model: meta-llama/llama-4-scout-17b-16e-instruct
- Current Gemini model: gemini-3.1-flash-lite-preview
- Prompt templates and model IDs are in vlm_config.json.

## 7) Data/Output Locations
- Input DICOM directory: CTImages/dicom_dir
- Generated volumes/manifests: generated/volumes
- Generated slice images: generated/slices
- TotalSeg outputs: generated/totalseg

## 8) Known Constraints
- AI output is triage support, not diagnostic ground truth.
- Live AI requests depend on valid API keys and model availability.

## 9) Recommended Next Steps
- Add user presets (for example: Bones only, Vessels only, Heart + Great Vessels, Lungs only).
- Add export/import for organ visibility presets.
- Sync README wording to match the current Gemini model id.
- Add basic automated frontend tests for organ visibility logic.
