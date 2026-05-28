from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "runs/detect/train-9/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image.")
    parser.add_argument("image_path", help="Path to the image to run inference on.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained YOLO weights file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.02,
        help="Confidence threshold for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    weights_path = Path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(source=str(image_path), save=True, conf=args.conf)

    for result in results:
        print(f"Saved prediction for: {result.path}")


if __name__ == "__main__":
    main()
