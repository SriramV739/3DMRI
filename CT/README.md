# CT Spatial VLM Viewer

This folder contains an end-to-end CT imaging prototype:

- FastAPI backend for DICOM ingestion, CT pseudo-coloring, 3D mesh export, and Groq VLM calls.
- TotalSegmentator pipeline for `data/raw/CT-chest.nrrd` anatomical segmentation and realistic GLB mesh generation.
- React + Three.js frontend with orbit controls and `@react-three/xr` AR/VR entry points for WebXR-capable browsers, including Apple Vision Pro Safari when WebXR is enabled by the device/browser.
- Native visionOS SwiftUI + RealityKit app in `VisionProDemo` for Apple Vision Pro simulator/device workflows.
- CT-local environment files only. Raw input stays in `CTImages`.

## Setup

From this `CT` directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend
npm install
```

Create your local API key file:

```bash
cp .env.example .env
```

Edit `CT/.env` so it contains:

```bash
GROQ_API_KEY=my_key
GEMINI_API_KEY=my_gemini_key
```

Without a provider key, `/api/analyze` returns a payload preview instead of calling the live VLM.

## Render Tests

Run the required one-scan smoke test:

```bash
.venv/bin/python backend/scripts/smoke_test.py
```

Run the required 10-render batch test:

```bash
.venv/bin/python backend/scripts/batch_test.py
```

Generated GLB volumes and JSON manifests are saved under `generated/volumes`. Pseudo-colored slice PNGs are saved under `generated/slices`.

Run the standalone rib-fracture stress test:

```bash
.venv/bin/python -m unittest tests.test_synthetic_rib_fracture_render -v
```

This test is independent of the app/backend rendering pipeline. It creates a synthetic CT-like rib cage with a known displaced, jagged left rib fracture, exports bone-only and annotated GLB meshes, and writes preview artifacts to:

- `generated/test_renders/synthetic_rib_fracture/realistic_left_rib_fracture_bone_only.glb`
- `generated/test_renders/synthetic_rib_fracture/realistic_left_rib_fracture_annotated.glb`
- `generated/test_renders/synthetic_rib_fracture/realistic_left_rib_fracture_bone_only_closeup.png`
- `generated/test_renders/synthetic_rib_fracture/realistic_left_rib_fracture_annotated_closeup.png`
- `generated/test_renders/synthetic_rib_fracture/synthetic_left_rib_fracture_projection.png`
- `generated/test_renders/synthetic_rib_fracture/synthetic_left_rib_fracture_metadata.json`

## TotalSegmentator Chest Render

The TotalSegmentator path uses:

```text
data/raw/CT-chest.nrrd
```

It converts the NRRD to NIfTI, runs TotalSegmentator, then exports a fresh anatomy-colored GLB. It does not use the old `data/output/meshes` folder.

Run the full regeneration:

```bash
.venv/bin/python backend/scripts/totalseg_chest.py --device mps --force-segmentation --fullres
```

Use CPU if MPS is unavailable:

```bash
.venv/bin/python backend/scripts/totalseg_chest.py --device cpu --force-segmentation --fullres
```

Rebuild only the GLB from the latest generated masks:

```bash
.venv/bin/python backend/scripts/totalseg_chest.py --skip-segmentation
```

Outputs:

- Converted NIfTI: `generated/totalseg/converted_inputs/CT-chest.nii.gz`
- Fresh masks: `generated/totalseg/segmentations`
- Viewer GLB: `generated/volumes/totalseg_CT_chest_realistic.glb`
- Viewer manifest: `generated/volumes/totalseg_CT_chest_realistic.json`

## Vision Pro Native App

The native app is in:

```text
VisionProDemo/CTVisionDemo.xcodeproj
```

It uses this runtime architecture:

```text
Vision Pro native app
  -> downloads a local-backend USDZ anatomy model
  -> shows grouped native organ visibility controls based on detected anatomy
  -> loads CT slice thumbnails from FastAPI
  -> captures the current 3D viewport and sends it with a typed question to Gemini
  -> sends selected slices plus a typed prompt to Gemini through FastAPI
```

Generate the Vision Pro optimized model from the TotalSegmentator render:

```bash
.venv/bin/python backend/scripts/export_visionos_assets.py --source-id totalseg_CT_chest_realistic --quality balanced
```

This creates:

- `generated/visionos/totalseg_CT_chest_realistic_visionos_balanced.usdz`
- `generated/visionos/totalseg_CT_chest_realistic_visionos_balanced.glb`
- `generated/visionos/totalseg_CT_chest_realistic_visionos_balanced.json`

The balanced asset is currently about 49 MB as USDZ, with 86 named anatomy meshes and about 629k vertices.

Build the visionOS simulator app from the `CT` folder:

```bash
xcodebuild -project VisionProDemo/CTVisionDemo.xcodeproj \
  -scheme CTVisionDemo \
  -configuration Debug \
  -sdk xrsimulator26.4 \
  -destination 'platform=visionOS Simulator,name=Apple Vision Pro' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Run it in Xcode:

1. Open `VisionProDemo/CTVisionDemo.xcodeproj`.
2. Select the Apple Vision Pro simulator or your paired Vision Pro.
3. Start the backend:

   ```bash
   .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. In the native app, use `http://127.0.0.1:8000` for the simulator. On a physical Vision Pro, use the Mac Wi-Fi IP, for example `http://192.168.1.23:8000`.
5. Press Connect. The app downloads the USDZ, shows grouped organ controls, displays slice thumbnails, and sends selected slices or the current 3D snapshot to Gemini.

For 3D model questions, use the `3D View Question` panel:

1. Rotate and zoom the model to the view you want to ask about.
2. Press `Capture`; the app freezes the 3D interaction state and stores the current viewport image.
3. Type your question and press `Ask Gemini`.
4. Press the unlock button to release the frozen view and continue exploring.

## Start Backend

```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Useful routes:

- `GET /api/health`
- `GET /api/slices`
- `POST /api/process?limit=10`
- `POST /api/process-totalseg`
- `GET /api/visionos/assets`
- `POST /api/visionos/export`
- `GET /api/volumes`
- `POST /api/analyze`
- `POST /api/gemini-test`
- `POST /api/analyze-form`
- `GET /docs`

## Start Frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

The frontend proxies `/api` and `/generated` to the backend at `http://127.0.0.1:8000`.

The TotalSegmentator chest render appears first in the volume dropdown after it has been generated.

## Open On Apple Vision Pro

The Vision Pro does not need to be physically connected to the Mac. It only needs to be on the same Wi-Fi network.

Recommended development mode:

1. On the Mac, start the backend on all network interfaces:

   ```bash
   cd CT
   .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. In a second terminal, start the frontend on all network interfaces:

   ```bash
   cd CT/frontend
   npm run dev
   ```

3. Find the Mac Wi-Fi IP address:

   ```bash
   ipconfig getifaddr en0
   ```

4. In Vision Pro Safari, open:

   ```text
   http://MAC_WIFI_IP:5173
   ```

   Example:

   ```text
   http://192.168.1.23:5173
   ```

The left side remains the interactive 3D CT volume with rotate, pan, zoom, organ visibility, and spatial entry when WebXR is available. The right side is the slice picker, text prompt, selected image strip, and VLM response panel.

Production-like single-server mode:

```bash
cd CT/frontend
npm run build
cd ..
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open this in Vision Pro Safari:

```text
http://MAC_WIFI_IP:8000
```

For immersive WebXR on Vision Pro, Safari must expose the WebXR Device API. If the Spatial button says WebXR is unavailable, open Safari settings on Vision Pro and enable the WebXR-related feature flags, then reload the page. The app still works as a windowed Vision Pro Safari app when immersive mode is unavailable.

## VLM Configuration

The VLM prompt and model are editable in `vlm_config.json`. Groq's current public vision documentation lists `meta-llama/llama-4-scout-17b-16e-instruct` as a supported image-input model. Qwen models currently listed on Groq are text-only, so this app defaults to Llama 4 Scout as the closest supported multimodal model. When Groq exposes a Qwen VL model ID, replace the `model` field in `vlm_config.json`.

Gemini is also available via `GEMINI_API_KEY` and defaults to `gemini-3.1-flash-lite-preview`. If Google returns a temporary high-demand/provider error for that preview model, the backend automatically tries the configured `gemini_fallback_models` so the Vision Pro app gets an analysis response instead of a generic 500.

The prompt asks the model to identify anatomy, summarize observations, flag potential abnormalities or patient issues, and separate uncertainty from findings. This is triage support only and is not a diagnostic substitute for radiologist review.

## Data Pipeline

The backend reads DICOM files from `CTImages/dicom_dir`, applies DICOM rescale slope/intercept to recover Hounsfield-like units, creates pseudo-colored slice previews, and exports GLB meshes with multiple isosurfaces:

- Lung/air boundary
- Soft tissue
- Contrast-enhanced tissue and dense structures
- Cortical bone

The dataset in this folder is mostly one CT slice per DICOM series. For true multi-slice series, the backend stacks slices by DICOM position/instance metadata. For single-slice cases, it creates a thin 3D slab so each scan can still be inspected as a spatial volume.

The TotalSegmentator pipeline is the higher-quality path for the full `CT-chest.nrrd` volume. It creates separate meshes for lungs, heart, vessels, ribs, vertebrae, sternum, clavicles, liver, spleen, kidneys, airway, and other detected anatomy, each with anatomy-specific color and transparency.
