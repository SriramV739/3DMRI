# CT Spatial VLM Viewer

This folder contains an end-to-end CT imaging prototype:

- FastAPI backend for DICOM ingestion, CT pseudo-coloring, 3D mesh export, and Groq VLM calls.
- TotalSegmentator pipeline for `data/raw/CT-chest.nrrd` anatomical segmentation and realistic GLB mesh generation.
- React + Three.js frontend with orbit controls and `@react-three/xr` AR/VR entry points for WebXR-capable browsers, including Apple Vision Pro Safari when WebXR is enabled by the device/browser.
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

## Start Backend

```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Useful routes:

- `GET /api/health`
- `GET /api/slices`
- `POST /api/process?limit=10`
- `POST /api/process-totalseg`
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

Gemini is also available via `GEMINI_API_KEY` and defaults to `gemini-2.5-flash-lite` to keep multimodal image-and-text calls on a lower-cost Gemini model.

The prompt asks the model to identify anatomy, summarize observations, flag potential abnormalities or patient issues, and separate uncertainty from findings. This is triage support only and is not a diagnostic substitute for radiologist review.

## Data Pipeline

The backend reads DICOM files from `CTImages/dicom_dir`, applies DICOM rescale slope/intercept to recover Hounsfield-like units, creates pseudo-colored slice previews, and exports GLB meshes with multiple isosurfaces:

- Lung/air boundary
- Soft tissue
- Contrast-enhanced tissue and dense structures
- Cortical bone

The dataset in this folder is mostly one CT slice per DICOM series. For true multi-slice series, the backend stacks slices by DICOM position/instance metadata. For single-slice cases, it creates a thin 3D slab so each scan can still be inspected as a spatial volume.

The TotalSegmentator pipeline is the higher-quality path for the full `CT-chest.nrrd` volume. It creates separate meshes for lungs, heart, vessels, ribs, vertebrae, sternum, clavicles, liver, spleen, kidneys, airway, and other detected anatomy, each with anatomy-specific color and transparency.
