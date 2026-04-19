# Phase 2: Live Intra-Op Augmented Reality Pipeline

Real-time laparoscopic AR overlay powered by a specialized detector plus MedSAM2 segmentation and NVIDIA Holoscan.

## Goal

The current proof-of-concept target is laparoscopic gallbladder surgery.

1. A specialized detector finds gallbladder-surgery anatomy and tools.
2. A VLM sidecar interprets the surgical intent query and chooses which anatomy prompts matter right now.
3. MedSAM2 refines those anatomy prompts into masks.
4. The overlay compositor renders color-coded AR guidance.
5. Holoviz or the offline evaluator displays the result.

The repo now supports two detector paths:

- `roboflow_hosted`
  Fastest POC path. Uses the public Roboflow Universe laparoscopy model.
- `local_yolo`
  Uses local YOLO weights. This can be a legacy generic model, a custom dual-model setup, or a fine-tuned POC model.

## Architecture

```text
Live path:
Video -> FormatConverter -> Detector backend -> VLM prompt guide -> Segmenter backend -> Overlay -> Holoviz

Offline evaluation path:
Staged laparoscopic videos -> batch conversion -> Detector backend -> VLM prompt guide -> MedSAM2 -> overlay + metrics
```

## Repository Layout

```text
surgery/
|-- config/app_config.yaml
|-- data/
|-- evaluation/
|-- models/
|-- operators/
|-- tests/
|-- app.py
`-- run.py
```

## Prerequisites

- Linux machine, ideally Ubuntu 22.04 or 24.04
- NVIDIA GPU with at least 8 GB VRAM
- NVIDIA drivers plus CUDA 12.1 or newer
- Python 3.10+
- Docker plus NVIDIA Container Toolkit for the Holoscan path

## Environment Setup

Run these commands from `surgery/` on the Linux GPU machine:

```bash
pip install -r requirements.txt
bash models/download_checkpoints.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected checkpoints:

- `models/MedSAM2_latest.pt`
- `models/yolov8n.pt`
- `models/sam2.1_hiera_tiny.pt`

## Proof-of-Concept Detector

The default detector backend is now `roboflow_hosted`.

Config:

- `detector.backend`
- `roboflow_laparoscopy.model_id`
- `roboflow_laparoscopy.api_url`
- `roboflow_laparoscopy.api_key_env`
- `roboflow_laparoscopy.confidence_threshold`
- `roboflow_laparoscopy.class_name_map`

As configured, the hosted POC model is:

- Roboflow Universe project: `laparoscopic-yolo/laparoscopy`
- dataset version: `14`
- model id: `laparoscopy/14`

The class mapping normalizes Roboflow labels like `Gallbladder`, `Duct`, and `Calot` into repo labels like `gallbladder`, `cystic_duct`, and `hepatocystic_triangle`.

If you want to avoid hosted inference latency, you can download the dataset and fine-tune a local YOLO segmentation model:

```bash
python scripts/download_roboflow_laparoscopy.py
python scripts/train_laparoscopy_poc.py
```

Then switch:

```yaml
detector:
  backend: "local_yolo"

yolo:
  model_path: "../models/laparoscopy_poc.pt"
```

## MedSAM2 Backend

The default segmentation backend is `medsam2`. The config exposes:

- `segmenter.backend`
- `medsam2.checkpoint`
- `medsam2.model_cfg`
- `medsam2.device`
- `medsam2.dtype`
- `medsam2.max_objects`
- `medsam2.use_temporal_memory`

The legacy `sam2` section remains available as a fallback backend.

## VLM Prompt Guide (Groq Llama 4 Scout)

The repo includes a VLM-guided anatomy prompt layer that interprets the surgeon's
natural-language query and filters which anatomy prompts are sent into MedSAM2.

## Scene Copilot

The repo now also includes a separate **lap-chole scene copilot** layer.

Its role is different from the prompt VLM:

- the prompt VLM decides what MedSAM2 should segment
- the scene copilot summarizes the scene, tracks workflow context over time, and answers conservative surgeon-facing questions

The intended runtime split is:

```text
Fast loop:
Detector -> Prompt VLM -> MedSAM2 -> Overlay

Slower loop:
Detector + masks + temporal state + user query -> Scene Copilot VLM -> structured scene analysis
```

The scene copilot returns structured analysis with:

- `scene_summary`
- `visible_structures`
- `visible_tools`
- `workflow_phase`
- `critical_view_status`
- `observed_risks`
- `uncertainties`
- `recommended_attention_targets`
- `qa_response`
- `confidence`

Config lives under `scene_copilot` in `config/app_config.yaml`.
The first version is intentionally conservative and lap-chole-specific.

### Recommended POC Path

The fastest path to a convincing live demo:

1. **Roboflow hosted detector** — zero training, instant bounding boxes
2. **Groq Llama 4 Scout VLM** — selects anatomy targets from the surgeon's query
3. **MedSAM2** — segments only the selected anatomy
4. **Holoscan** — orchestrates and displays the result in real time

### Groq Setup

1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Export it:

```bash
export GROQ_API_KEY="<your_groq_api_key>"
```

3. The config is already set up in `config/app_config.yaml`:

```yaml
vlm:
  enabled: true
  provider: "openai_compatible"
  api_url: "https://api.groq.com/openai/v1/chat/completions"
  api_key_env: "GROQ_API_KEY"
  model: "meta-llama/llama-4-scout-17b-16e-instruct"
```

### Config Reference

- `vlm.enabled` — toggle the VLM layer on/off
- `vlm.provider` — `"openai_compatible"` (Groq) or `"rule_based"` (offline fallback)
- `vlm.user_query` — the surgeon's natural-language intent
- `vlm.candidate_labels` — anatomy labels the VLM can select from
- `vlm.anatomy_aliases` — alternative names for each label
- `vlm.prompt_every_n_frames` — how often to re-query the VLM (default: 30)
- `vlm.max_image_size` — max dimension for the frame sent to the VLM (default: 512px; keep low to stay under Groq's 4 MB base64 limit)
- `vlm.api_url` — chat completions endpoint
- `vlm.api_key_env` — environment variable holding the API key
- `vlm.model` — model identifier

### Providers

- **`openai_compatible`** (default)
  Production-ready for Groq Llama 4 Scout. Sends the current frame, detections,
  and surgical query to the VLM and expects `target_labels` + `rationale` JSON back.
  Includes retry logic with exponential backoff for rate limits (HTTP 429),
  base64 payload size validation, and markdown fence stripping.
  Also works with any OpenAI-compatible vision-chat endpoint.

- **`rule_based`**
  Offline fallback. Maps the query to anatomy labels using keyword/alias matching.
  No API key required — useful for development and CI.

### Example Queries

- "show only the cystic duct and gallbladder"
- "focus on Calot's triangle"
- "segment the liver boundary and ignore tools"
- "highlight everything during the critical view of safety"

## Interactive Chat UI

The repo now includes a Streamlit front end for single-image interactive VLM prompting:

```text
Image -> Detector -> Groq prompt VLM -> MedSAM2 -> Overlay
     \-> Groq scene copilot -> side panel + chat response
```

Use it when you want to iterate quickly on prompts like:

- "highlight the liver"
- "now highlight the surgical tools instead"
- "focus only on the gallbladder and cystic duct"

Launch it from `surgery/`:

```bash
streamlit run scripts/interactive_vlm.py
```

What it does:

- caches the detector, VLM guide, segmenter, and overlay objects with `st.cache_resource`
- caches the scene copilot alongside the visual pipeline
- lets you pick an image from `data/clean.v5i.yolov8/test/images`
- also accepts uploaded laparoscopic frames
- preserves chat history per selected image
- keeps segmentation prompts separate from scene-analysis questions
- shows a side panel with scene summary, workflow phase, risks, uncertainties, and attention targets

Notes:

- The app uses the same `config/app_config.yaml` as the rest of the repo.
- For Groq-backed prompting, set `GROQ_API_KEY` in your environment.
- For hosted Roboflow detection, set `ROBOFLOW_API_KEY` in your environment if needed.

## Local Video Staging

For POC evaluation, stage real laparoscopic videos like this:

```text
data/cholec80/raw/video01.mp4
data/cholec80/raw/video02.mp4
data/cholec80/raw/video03.mp4
```

Optional evaluation-only seed prompts can be supplied as:

```text
data/cholec80/prompts/video01.json
data/cholec80/prompts/video02.json
data/cholec80/prompts/video03.json
```

Example prompt file:

```json
{
  "detections_by_frame": {
    "0": [
      {
        "class_name": "gallbladder",
        "bbox": [120, 80, 260, 210],
        "confidence": 1.0,
        "class_id": 0
      }
    ]
  }
}
```

## Offline Evaluation

The evaluator ingests staged videos, extracts JPEG frames for MedSAM2 video inference, saves a per-video bundle, and runs detector -> MedSAM2 -> overlay offline.

Run the default bounded evaluation:

```bash
python run.py \
  --config config/app_config.yaml \
  --video-dir data/cholec80/raw \
  --video-glob "video*.mp4" \
  --output-dir data/cholec80/processed \
  --max-frames 600 \
  --save-overlays \
  --save-masks
```

Outputs per video:

- `overlay.mp4`
- sampled overlay PNG frames
- per-frame mask NPZ files if `--save-masks` is set
- `metrics.json`

Top-level output:

- `data/cholec80/processed/manifest.json`
- `data/cholec80/processed/summary.json`

Recorded metrics include:

- frames processed
- frames with detector detections
- frames with prompts
- frames with non-empty masks
- average masks per frame
- median and p95 latency
- peak VRAM
- number of frames that used seed prompts

## Hosted Detector Notes

Hosted Roboflow inference is the fastest path to a convincing demo, but it depends on:

- internet connectivity
- Roboflow API availability
- per-frame network latency

For a polished live demo, the next step is to replace the hosted detector with the locally fine-tuned `laparoscopy_poc.pt` model.

## Batch Video Conversion Only

If you only want to ingest videos into staged assets:

```bash
python data/convert_video.py \
  --video-dir data/cholec80/raw \
  --video-glob "video*.mp4" \
  --output data/cholec80/processed \
  --max-frames 600 \
  --save-frames
```

Each video is written to its own folder:

```text
data/cholec80/processed/video01/
|-- video01.gxf_entities.npy
|-- video01_metadata.json
`-- frames/00000.jpg ...
```

## Holoscan App

To run the live application:

```bash
docker-compose build
docker-compose up
```

Or, if Holoscan is installed directly:

```bash
python run.py --config config/app_config.yaml
```

## Smoke Tests

Existing smoke tests are still available:

```bash
pytest tests/test_format_utils.py -v
pytest tests/test_yolo_detection_op.py -v -s
pytest tests/test_overlay_compositor.py -v -s
pytest tests/test_sam2_inference_op.py -v -s
pytest tests/test_medsam2_inference_op.py -v -s
pytest tests/test_yolo_sam_integration.py -v -s
pytest tests/test_full_pipeline.py -v -s
pytest tests/test_convert_video_batch.py -v
pytest tests/test_offline_evaluator.py -v
```

Quick local helper:

```bash
python run.py --smoke-test
```

## Notes

- MedSAM2 upstream repo: https://github.com/bowang-lab/MedSAM2
- Roboflow Universe laparoscopy model: https://universe.roboflow.com/laparoscopic-yolo/laparoscopy
- The offline evaluator is primarily qualitative plus runtime/VRAM validation; it does not compute Dice or IoU labels.
