import cv2
import os
import numpy as np


def get_dot_centers(mask: np.ndarray, min_area: int = 3, max_area:int = 80):
    """
    Found the center coordinates of each dot in a black-and-white mask image.
    Filtered out circles that are too small, too large, or not circular enough to be a real dot.

    Args:
        mask (np.ndarray): A black and white mask of the image
        min_area (int, optional): The minimum area of a contour to be considered a dot. Defaults to 3.
        max_area (int, optional): The maximum area of a contour to be considered a dot. Defaults to 80.

    Returns:
        list[tuple[int, int]]: A list of (x, y) coordinates representing the centers of the detected dots
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for c in contours:
        area = cv2.contourArea(c)
        
        if not (min_area < area < max_area):
            continue
        perimeter = cv2.arcLength(c, True)
        
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity > 0.5:
            M = cv2.moments(c)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers


def estimate_tree_box(image: np.ndarray, cx: int, cy: int) -> tuple[int, int, int, int]:
    """
    Estimated a bounding box around a tree given its dot center. Size of the box is
    based on where the dot is vertically: dots lower in the image are closer
    to the camera so they get a bigger box, dots higher up are farther away
    so they get a smaller one.

    Args:
        image (np.ndarray): original image to get dimensions from
        cx (int): x coordinate of the center of the dot
        cy (int): y coordinate of the center of the dot

    Returns:
        tuple[int, int, int, int]:the bounding box as (x1, y1, x2, y2) where
            (x1, y1) is the top left corner and (x2, y2) is the bottom right corner
    """
    h, w = image.shape[:2]
    vertical_position = cy / h
    box_size = int(150 + (400 - 150) * vertical_position)
    half = box_size // 2
    
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(w, cx + half),
        min(h, cy + half)
    )

def bbox_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Converts a bounding box in pixel coordinates to YOLO format (normalized).

    Args:
        x1, y1 (int): top-left corner of the bounding box
        x2, y2 (int): bottom-right corner of the bounding box
        img_w (int): image width in pixels
        img_h (int): image height in pixels

    Returns:
        tuple[float, float, float, float]: (x_center, y_center, width, height) normalized 0-1
    """
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

images_dir = "../forest_images"
output_dir = "output_labels"
# Check if output folder exists, if not create it
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    if not filename.endswith(".jpg"):
        continue
    image_path = f"{images_dir}/{filename}"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}, skipping.")
        continue

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Image's height and width for normalizaiton of bounding box coordinates
    img_h, img_w = image.shape[:2]

    # Red wraps in HSV so we need two ranges
    red_mask = (
        cv2.inRange(hsv, np.array([0, 200, 200]), np.array([5, 255, 255])) |
        cv2.inRange(hsv, np.array([175, 200, 200]), np.array([180, 255, 255]))
    )

    # Green dot HSV values sampled from real images: H~48, S~200-225, V~185-190
    green_mask = cv2.inRange(hsv, np.array([44, 180, 150]), np.array([55, 235, 210]))

    red_centers = get_dot_centers(red_mask)
    green_centers = get_dot_centers(green_mask)

    print(f"{image_path}: {len(red_centers)} safe trees (Class 0), {len(green_centers)} unsafe trees (Class 1).")

    lines = []

    for cx, cy in red_centers:
        x1, y1, x2, y2 = estimate_tree_box(image, cx, cy)
        xc, yc, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"0 {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}  # Safe Tree (Class 0)")

    for cx, cy in green_centers:
        x1, y1, x2, y2 = estimate_tree_box(image, cx, cy)
        xc, yc, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"1 {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}  # Unsafe Tree (Class 1)")

    stem = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    txt_path = f"{output_dir}/{stem}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

# Check
    print(f"Saved {txt_path}")
