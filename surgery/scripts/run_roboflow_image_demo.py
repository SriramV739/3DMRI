"""
Run the Roboflow laparoscopy model on a single image and save an annotated preview.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont


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


def polygon_from_points(points):
    return [(float(point["x"]), float(point["y"])) for point in points]


def draw_prediction(draw: ImageDraw.ImageDraw, prediction: dict, font):
    label = prediction["class"]
    color = DEFAULT_COLORS.get(label, (255, 255, 255))
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])
    x1 = x - width / 2.0
    y1 = y - height / 2.0
    x2 = x + width / 2.0
    y2 = y + height / 2.0

    points = prediction.get("points") or []
    if points:
        polygon = polygon_from_points(points)
        draw.polygon(polygon, outline=color, width=3)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=6, outline=color, width=2)

    text = f"{label} {prediction.get('confidence', 0.0):.2f}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx1 = max(0, x1)
    ty1 = max(0, y1 - th - 6)
    tx2 = tx1 + tw + 8
    ty2 = ty1 + th + 4
    draw.rounded_rectangle((tx1, ty1, tx2, ty2), radius=4, fill=(0, 0, 0), outline=color, width=2)
    draw.text((tx1 + 4, ty1 + 2), text, fill=(255, 255, 255), font=font)


def main():
    parser = argparse.ArgumentParser(description="Run the Roboflow laparoscopy model on a single image")
    parser.add_argument("--image", default="data/roboflow_demo/example_thumb.jpg")
    parser.add_argument("--output", default="preview_images/roboflow_example_predicted.png")
    parser.add_argument("--model-id", default="laparoscopy/14")
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    args = parser.parse_args()

    api_key = args.api_key if args.api_key is not None else os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing Roboflow API key. Set {args.api_key_env} or pass --api-key.")

    client = InferenceHTTPClient(api_url=args.api_url, api_key=api_key)
    result = client.infer(args.image, model_id=args.model_id)

    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    predictions = sorted(result.get("predictions", []), key=lambda pred: pred.get("confidence", 0.0), reverse=True)
    for prediction in predictions:
        draw_prediction(draw, prediction, font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(output_path.resolve())
    print(f"predictions={len(predictions)}")
    for prediction in predictions:
        print(f"{prediction['class']}: {prediction.get('confidence', 0.0):.3f}")


if __name__ == "__main__":
    main()
