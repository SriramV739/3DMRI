"""
Run the Roboflow laparoscopy model on a video and save an annotated preview clip.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from inference_sdk import InferenceHTTPClient


DEFAULT_COLORS = {
    "Gallbladder": (0, 255, 128),
    "Duct": (255, 165, 0),
    "Calot": (255, 215, 0),
    "Liver": (0, 220, 120),
    "Forceps": (100, 149, 237),
    "Allis": (100, 149, 237),
    "Cautery": (255, 140, 0),
    "Clipper": (255, 80, 80),
    "Suction": (120, 180, 255),
    "Bag": (180, 100, 255),
    "scissors": (240, 240, 240),
}


def draw_prediction(frame_bgr, prediction: dict):
    label = prediction["class"]
    color = DEFAULT_COLORS.get(label, (255, 255, 255))
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])
    x1 = int(round(x - width / 2.0))
    y1 = int(round(y - height / 2.0))
    x2 = int(round(x + width / 2.0))
    y2 = int(round(y + height / 2.0))

    points = prediction.get("points") or []
    if points:
        polygon = [
            [int(round(point["x"])), int(round(point["y"]))]
            for point in points
        ]
        if len(polygon) >= 3:
            cv2.polylines(
                frame_bgr,
                [np.array(polygon, dtype=np.int32)],
                isClosed=True,
                color=color,
                thickness=2,
            )

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    caption = f"{label} {prediction.get('confidence', 0.0):.2f}"
    (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty1 = max(0, y1 - th - baseline - 6)
    ty2 = ty1 + th + baseline + 6
    tx2 = min(frame_bgr.shape[1] - 1, x1 + tw + 8)
    cv2.rectangle(frame_bgr, (x1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        frame_bgr,
        caption,
        (x1 + 4, ty2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def infer_frame(client: InferenceHTTPClient, frame_bgr, model_id: str):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        temp_path = handle.name
    try:
        success = cv2.imwrite(temp_path, frame_bgr)
        if not success:
            raise RuntimeError("Failed to encode frame for inference")
        return client.infer(temp_path, model_id=model_id)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run Roboflow laparoscopy inference on a video clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="preview_images/roboflow_video_demo.mp4")
    parser.add_argument("--samples-dir", default="preview_images/video_samples")
    parser.add_argument("--model-id", default="laparoscopy/14")
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--max-frames", type=int, default=45)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    api_key = args.api_key if args.api_key is not None else os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing Roboflow API key. Set {args.api_key_env} or pass --api-key.")

    client = InferenceHTTPClient(api_url=args.api_url, api_key=api_key)
    input_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(input_path))
    metadata = reader.get_meta_data()
    src_fps = float(metadata.get("fps") or 25.0)
    if args.start_frame < 0:
        raise RuntimeError("--start-frame must be >= 0")

    first_frame_rgb = None
    for _ in range(args.start_frame + 1):
        try:
            first_frame_rgb = reader.get_next_data()
        except IndexError as exc:
            raise RuntimeError("start frame is beyond the end of the video") from exc
    if first_frame_rgb is None:
        raise RuntimeError("Unable to read starting frame from the video")
    src_height, src_width = first_frame_rgb.shape[:2]

    scale = args.target_width / float(src_width)
    out_width = args.target_width
    out_height = int(round(src_height * scale))
    writer = imageio.get_writer(
        str(output_path),
        fps=max(1.0, src_fps / max(1, args.stride)),
        codec="libx264",
        quality=7,
    )

    frame_idx = args.start_frame
    written = 0
    try:
        current_frame_rgb = first_frame_rgb
        while written < args.max_frames:
            if frame_idx > args.start_frame:
                try:
                    current_frame_rgb = reader.get_next_data()
                except IndexError:
                    break

            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            frame_bgr = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(frame_bgr, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
            result = infer_frame(client, resized, args.model_id)
            predictions = sorted(
                result.get("predictions", []),
                key=lambda prediction: prediction.get("confidence", 0.0),
                reverse=True,
            )
            for prediction in predictions:
                draw_prediction(resized, prediction)

            writer.append_data(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            if written % args.sample_every == 0:
                cv2.imwrite(str(samples_dir / f"frame_{written:03d}.png"), resized)
            print(f"frame={written} predictions={len(predictions)}")
            written += 1
            frame_idx += 1
    finally:
        reader.close()
        writer.close()

    print(output_path.resolve())
    print(samples_dir.resolve())


if __name__ == "__main__":
    main()
