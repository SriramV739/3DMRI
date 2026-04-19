#!/bin/bash
# Download YOLO, MedSAM2, and fallback SAM2 checkpoints.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)"

download_file() {
  local url="$1"
  local output="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
  else
    echo "[error] Install wget or curl to download model checkpoints."
    exit 1
  fi
}

echo "[info] Downloading MedSAM2_latest.pt..."
download_file \
  "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt" \
  "$MODELS_DIR/MedSAM2_latest.pt"

echo "[info] Downloading fallback SAM2.1 tiny checkpoint..."
download_file \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
  "$MODELS_DIR/sam2.1_hiera_tiny.pt"

echo "[info] Downloading YOLOv8n checkpoint (~6.3MB)..."
python3 -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
print('YOLOv8n loaded successfully')
" 2>/dev/null

YOLO_CACHE=$(find ~/.config/Ultralytics -name "yolov8n.pt" 2>/dev/null | head -1 || true)
if [ -z "$YOLO_CACHE" ]; then
  YOLO_CACHE=$(find ~/. -name "yolov8n.pt" -path "*/ultralytics/*" 2>/dev/null | head -1 || true)
fi

if [ -n "$YOLO_CACHE" ]; then
  cp "$YOLO_CACHE" "$MODELS_DIR/yolov8n.pt"
else
  echo "[warn] YOLOv8n was downloaded by ultralytics but not found in cache."
  echo "       Runtime will fall back to ultralytics default model lookup."
fi

echo
echo "[ok] Model checkpoints ready:"
ls -lh "$MODELS_DIR"/*.pt 2>/dev/null || echo "  (check paths above)"
