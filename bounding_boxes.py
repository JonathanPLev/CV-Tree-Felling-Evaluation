import cv2
import json
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


def estimate_tree_box(image: NDArray[np.uint8], cx: int, cy: int) -> tuple[int, int, int, int]:
    """
    Estimated a bounding box around a tree given its dot center. Size of the box is
    based on where the dot is vertically: dots lower in the image are closer
    to the camera so they get a bigger box, dots higher up are farther away
    so they get a smaller one.

    Args:
        image (NDArray[np.uint8]): original image to get dimensions from
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


def is_surrounded_by_bark(hsv_image: NDArray[np.uint8], cx: int, cy: int, ring_radius: int = 10) -> bool:
    """
    Trying to determine if the dot is valid or not. Validating by sampling pixels around the circle to see if it is bark.
    This threw out the false positives.
    

    Args:
        hsv_image (NDArray[np.uint8]): image converted to HSV color space
        cx (int): x coordinate of the center of the dot
        cy (int): y coordinate of the center of the dot
        ring_radius (int, optional): _description_. Defaults to 10.

    Returns:
        bool: True if surrounding pixels look like bark, False otherwise
    """
    h, w = hsv_image.shape[:2]
    bark_count = 0
    total = 0
    
    for angle in range(0, 360, 10):
        rad = np.deg2rad(angle)
        x = int(cx + ring_radius * np.cos(rad))
        y = int(cy + ring_radius * np.sin(rad))
        
        if 0 <= x < w and 0 <= y < h:
            hue, sat, val = hsv_image[y, x]
            
            if sat < 50 and 50 < val < 180:
                bark_count += 1
            total += 1

    return (bark_count / total) > 0.6 if total > 0 else False

# Testing one image
image = cv2.imread("IMG_1943.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red wraps in HSV so we need two ranges
red_mask = (
    cv2.inRange(hsv, np.array([0, 200, 200]), np.array([5, 255, 255])) |
    cv2.inRange(hsv, np.array([175, 200, 200]), np.array([180, 255, 255]))
)

green_mask = cv2.inRange(hsv, np.array([42, 150, 150]), np.array([55, 255, 255]))
green_mask = cv2.dilate(green_mask, np.ones((3, 3), np.uint8), iterations=1)

red_centers = get_dot_centers(red_mask)
green_centers = [
    (cx, cy) for (cx, cy) in get_dot_centers(green_mask)
    if is_surrounded_by_bark(hsv, cx, cy)
]

print(f"Found {len(red_centers)} healthy trees, {len(green_centers)} trees to cut down.")

annotations = []
output_image = image.copy()

for i, (cx, cy) in enumerate(red_centers):
    x1, y1, x2, y2 = estimate_tree_box(image, cx, cy)
    annotations.append({"tree_id": i + 1, "label": "healthy", "bbox": [x1, y1, x2, y2]})
    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

for i, (cx, cy) in enumerate(green_centers):
    x1, y1, x2, y2 = estimate_tree_box(image, cx, cy)
    annotations.append({"tree_id": len(red_centers) + i + 1, "label": "diseased", "bbox": [x1, y1, x2, y2]})
    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("output_boxes.jpg", output_image)
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)

# print("Saved output_boxes.jpg and annotations.json")