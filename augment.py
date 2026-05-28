import os
import cv2
import albumentations as A

# #Config
<<<<<<< Updated upstream
<<<<<<< Updated upstream
IMAGE_DIR = "images/original_images"
LABEL_DIR = "labels/output_labels"
OUTPUT_DIR = "images/augmented_images"
LABEL_OUTPUT_DIR = "labels/augmented_labels"
IMAGE_NAMES = ["IMG_1943", "IMG_1944", "IMG_1945", "IMG_1946", "IMG_1947"]
=======
=======
>>>>>>> Stashed changes
IMAGE_DIR = "ml/datasets/images/train"
LABEL_DIR = "ml/datasets/labels/train"
OUTPUT_DIR = "images/augmented_images-1"
LABEL_OUTPUT_DIR = "labels/augmented_labels-1"
IMAGE_NAMES = ["IMG_1943", "IMG_1944", "IMG_1946", "IMG_1947"] # hold out IMG_1945 for now
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
IMG_EXT = ".jpg"

# 10 augmentation pipelines
# I used a baseline min_visibility=0.3 for all augmentations, not sure if we want to change that for some of them
# clipped the boxes to the boundaries
AUGMENTATIONS = [
    # Horizontal flip
    A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Brightness increase
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=0, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Brightness decrease
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.3), contrast_limit=0, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Random crop + resize back
    A.Compose([
        A.RandomResizedCrop(size=(1024, 768), scale=(0.75, 0.90), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Slight rotation clockwise
    A.Compose([
        A.Rotate(limit=(10, 10), border_mode=cv2.BORDER_REFLECT, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Slight rotation counter-clockwise
    A.Compose([
        A.Rotate(limit=(-10, -10), border_mode=cv2.BORDER_REFLECT, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Simulated shadow
    A.Compose([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Contrast boost
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.3, 0.3), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # HSV color jitter
    # it shifts the greens and browns, which is good for forest scenes
    A.Compose([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),

    # Slight gaussian blur
    # I used a blur_limit of 5-7, we can change this as well
    A.Compose([
        A.GaussianBlur(blur_limit=(5, 7), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)),
]

AUG_NAMES = [
    "hflip", "bright_up", "bright_down", "crop_resize",
    "rotate_cw", "rotate_ccw", "shadow", "contrast",
    "hsv_jitter", "blur"
]

# read the yolo txt file
def read_yolo_labels(txt_path):
    classes, bboxes = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # clamp to valid range to handle floating point issues in labels
            # ^ran into this error for images 1945 and 1946
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            classes.append(cls)
            bboxes.append([x_c, y_c, w, h])
    return classes, bboxes

# write new yolo txt files
def write_yolo_labels(txt_path, classes, bboxes):
    label_map = {0: "Safe Tree (Class 0)", 1: "Unsafe Tree (Class 1)"}
    with open(txt_path, "w") as f:
        for cls, bbox in zip(classes, bboxes):
            x_c, y_c, w, h = bbox
            comment = label_map.get(cls, f"Class {cls}")
            f.write(f"{cls} {x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}  # {comment}\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)
    total_generated = 0
    for name in IMAGE_NAMES:
        img_path = os.path.join(IMAGE_DIR, name + IMG_EXT)
        txt_path = os.path.join(LABEL_DIR, name + ".txt")
        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue
        if not os.path.exists(txt_path):
            print(f"[SKIP] Labels not found: {txt_path}")
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classes, bboxes = read_yolo_labels(txt_path)
        print(f"\nProcessing {name} ({len(bboxes)} boxes, image size {image.shape[1]}x{image.shape[0]})")

        for i, (aug, aug_name) in enumerate(zip(AUGMENTATIONS, AUG_NAMES)):
            try:
                result = aug(image=image, bboxes=bboxes, class_labels=classes)
            except Exception as e:
                print(f"[ERROR] aug {i} ({aug_name}): {e}")
                continue

            aug_image = result["image"]
            aug_bboxes = result["bboxes"]
            aug_classes = result["class_labels"]
            out_name = f"{name}_aug_{i}_{aug_name}"
            out_img = os.path.join(OUTPUT_DIR, out_name + IMG_EXT)
            out_txt = os.path.join(LABEL_OUTPUT_DIR, out_name + ".txt")

            cv2.imwrite(out_img, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_labels(out_txt, aug_classes, aug_bboxes)
            print(f"[{i}] {aug_name:15s} -> {len(aug_bboxes)} boxes kept -> {out_name}")
            total_generated += 1

    print(f"\n{total_generated} augmented images generated./'")

if __name__ == "__main__":
    main()
