#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bounding_boxes import get_dot_centers


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove human-annotated red/green dots from images using OpenCV color masks "
            "and inpainting."
        )
    )
    parser.add_argument(
        "--images-dir",
        default="ml/datasets/images/train",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--labels-dir",
        default="ml/datasets/labels/train",
        help="Directory containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/datasets/images/train",
        help="Directory where cleaned images will be saved.",
    )
    parser.add_argument(
        "--mask-radius",
        type=int,
        default=8,
        help="Circle radius in pixels for each detected annotation center.",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=float,
        default=3.0,
        help="Inpainting neighborhood radius passed to cv2.inpaint.",
    )
    return parser.parse_args()


def detect_dot_centers(image: np.ndarray) -> List[Tuple[int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = (
        cv2.inRange(hsv, np.array([0, 200, 200]), np.array([5, 255, 255]))
        | cv2.inRange(hsv, np.array([175, 200, 200]), np.array([180, 255, 255]))
    )
    green_mask = cv2.inRange(hsv, np.array([44, 180, 150]), np.array([55, 235, 210]))

    red_centers = get_dot_centers(red_mask)
    green_centers = get_dot_centers(green_mask)
    return red_centers + green_centers


def inpaint_dots(image: np.ndarray, centers: List[Tuple[int, int]], mask_radius: int, inpaint_radius: float) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x, y in centers:
        cv2.circle(mask, (x, y), mask_radius, 255, thickness=-1)
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"[WARN] Missing label for image: {image_path.name}")
    return pairs


def process_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    mask_radius: int,
    inpaint_radius: float,
) -> None:
    if mask_radius <= 0:
        raise ValueError("mask_radius must be > 0.")
    if inpaint_radius <= 0:
        raise ValueError("inpaint_radius must be > 0.")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_image_label_pairs(images_dir, labels_dir)

    if not pairs:
        print("[INFO] No matching image/label pairs found.")
        return

    print(f"[INFO] Processing {len(pairs)} matching image/label pairs.")
    success = 0
    skipped = 0

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] [{index}/{len(pairs)}] Could not read image: {image_path}")
                skipped += 1
                continue

            h, w = image.shape[:2]
            centers = detect_dot_centers(image)
            cleaned = inpaint_dots(image, centers, mask_radius=mask_radius, inpaint_radius=inpaint_radius)

            output_path = output_dir / image_path.name
            if not cv2.imwrite(str(output_path), cleaned):
                print(f"[WARN] [{index}/{len(pairs)}] Failed to write output: {output_path}")
                skipped += 1
                continue

            success += 1
            print(
                f"[OK] [{index}/{len(pairs)}] {image_path.name} cleaned "
                f"(points={len(centers)}, size={w}x{h}) -> {output_path}"
            )
        except FileNotFoundError as exc:
            print(f"[WARN] [{index}/{len(pairs)}] Missing file: {exc}")
            skipped += 1
        except Exception as exc:  # defensive fail-safe for batch jobs
            print(f"[WARN] [{index}/{len(pairs)}] Failed {image_path.name}: {exc}")
            skipped += 1

    print(f"[DONE] Cleaned: {success}, Skipped: {skipped}, Total pairs: {len(pairs)}")


def main() -> None:
    args = parse_args()
    process_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        mask_radius=args.mask_radius,
        inpaint_radius=args.inpaint_radius,
    )


if __name__ == "__main__":
    main()
