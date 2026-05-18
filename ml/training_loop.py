from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml
from ultralytics import YOLO


DATA_YAML = "forest_safety.yaml"
MODEL_NAME = "yolov8n.pt"
TARGET_CLASS_NAME = os.getenv("TARGET_CLASS_NAME", "Unsafe Tree")
TARGET_CLASS_INDEX = os.getenv("TARGET_CLASS_INDEX", "1")


@dataclass
class DecisionMetric:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, iw: int, ih: int) -> Tuple[float, float, float, float]:
    bw = w * iw
    bh = h * ih
    x = xc * iw
    y = yc * ih
    return x - bw / 2.0, y - bh / 2.0, x + bw / 2.0, y + bh / 2.0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _load_data_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_val_images(data: Dict, data_yaml_path: str) -> List[Path]:
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (Path(data_yaml_path).parent / root).resolve()

    val = data["val"]
    val_path = Path(val)
    if not val_path.is_absolute():
        val_path = (root / val).resolve()

    if val_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted([p for p in val_path.rglob("*") if p.suffix.lower() in exts])

    if val_path.is_file():
        if val_path.suffix.lower() == ".txt":
            items = []
            for line in val_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (val_path.parent / p).resolve()
                items.append(p)
            return items
        return [val_path]

    raise FileNotFoundError(f"Unable to resolve validation images from: {val_path}")


def _label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _get_target_class_index(data: Dict) -> int:
    if TARGET_CLASS_INDEX is not None:
        return int(TARGET_CLASS_INDEX)

    names = data.get("names", {})
    if isinstance(names, list):
        if TARGET_CLASS_NAME in names:
            return names.index(TARGET_CLASS_NAME)
    elif isinstance(names, dict):
        for k, v in names.items():
            if v == TARGET_CLASS_NAME:
                return int(k)

    raise ValueError(
        "Could not infer target class index. Set TARGET_CLASS_INDEX env var or ensure "
        f"TARGET_CLASS_NAME='{TARGET_CLASS_NAME}' exists in dataset names."
    )


def _load_gt_for_image(label_path: Path, iw: int, ih: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []

    gt = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row:
            continue
        cls_str, xc, yc, w, h = row.split()[:5]
        cls = int(float(cls_str))
        box = _xywhn_to_xyxy(float(xc), float(yc), float(w), float(h), iw, ih)
        gt.append((cls, box))
    return gt


def evaluate_decision_metric(model: YOLO, data_yaml: str, iou_match: float = 0.1) -> DecisionMetric:
    data = _load_data_yaml(data_yaml)
    target_cls = _get_target_class_index(data)
    images = _resolve_val_images(data, data_yaml)

    tp = fp = fn = 0

    for result in model.predict(source=[str(p) for p in images], stream=True, conf=0.001, verbose=False):
        image_path = Path(result.path)
        h, w = result.orig_shape
        gts = _load_gt_for_image(_label_path_from_image(image_path), w, h)

        pred_boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        pred_cls = [int(c) for c in (result.boxes.cls.cpu().tolist() if result.boxes is not None else [])]

        matched_pred = set()
        for gt_cls, gt_box in gts:
            best_idx, best_iou = -1, 0.0
            for i, pbox in enumerate(pred_boxes):
                if i in matched_pred:
                    continue
                score = _iou(gt_box, pbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_idx >= 0 and best_iou >= iou_match:
                matched_pred.add(best_idx)
                if gt_cls == target_cls and pred_cls[best_idx] == target_cls:
                    tp += 1
                elif gt_cls != target_cls and pred_cls[best_idx] == target_cls:
                    fp += 1
                elif gt_cls == target_cls and pred_cls[best_idx] != target_cls:
                    fn += 1
            else:
                if gt_cls == target_cls:
                    fn += 1

        for i, cls in enumerate(pred_cls):
            if i not in matched_pred and cls == target_cls:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DecisionMetric(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def main() -> None:
    model = YOLO(MODEL_NAME)

    # Step 2: prioritize classification correctness over localization.
    train_args = dict(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=8,
        patience=30,
        device=0,
        pretrained=True,
        freeze=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        close_mosaic=10,
        cls=1.8,
        box=5.0,
        plots=True,
    )

    model.train(**train_args)

    # Step 1: choose checkpoint by target-class decision metric, not default detector fitness.
    best_ckpt = model.trainer.best
    last_ckpt = model.trainer.last

    best_model = YOLO(str(best_ckpt))
    last_model = YOLO(str(last_ckpt))

    best_metric = evaluate_decision_metric(best_model, DATA_YAML)
    last_metric = evaluate_decision_metric(last_model, DATA_YAML)

    selected_model = best_model if best_metric.f1 >= last_metric.f1 else last_model
    selected_name = "best" if selected_model is best_model else "last"
    selected_metric = best_metric if selected_model is best_model else last_metric

    print(
        f"Selected checkpoint={selected_name} using target-class F1={selected_metric.f1:.4f} "
        f"(P={selected_metric.precision:.4f}, R={selected_metric.recall:.4f}, "
        f"TP={selected_metric.tp}, FP={selected_metric.fp}, FN={selected_metric.fn})"
    )

    # Step 3: tune around classification-sensitive hyperparameters.
    tune_space = {
        "lr0": (1e-5, 2e-3),
        "lrf": (0.005, 0.05),
        "momentum": (0.85, 0.98),
        "weight_decay": (1e-5, 1e-3),
        "cls": (1.2, 3.0),
        "box": (3.0, 6.0),
        "hsv_h": (0.0, 0.02),
        "hsv_s": (0.2, 0.6),
        "hsv_v": (0.1, 0.4),
        "scale": (0.05, 0.3),
        "fliplr": (0.0, 0.5),
    }

    selected_model.tune(
        data=DATA_YAML,
        epochs=80,
        iterations=30,
        optimizer="AdamW",
        imgsz=640,
        batch=8,
        space=tune_space,
        plots=True,
        val=True,
    )

    selected_model.export(format="coreml", imgsz=640, nms=True, half=True)


if __name__ == "__main__":
    main()
