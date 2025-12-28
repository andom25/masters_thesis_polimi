"""
This script implements an interactive image alignment and region extraction system for processing pairs of images from
two folders. The system enables manual alignment of a second image (Netfabb slice) over a first image (buildplate)
through user interaction, allowing position and scale adjustments. After alignment, the script extracts specific regions
defined in a JSON file, applies masks to identify infill areas (non-contour, non-white regions), and saves the cropped
regions as RGBA images with transparency. The implementation uses OpenCV for image processing, HSV color space for
green contour detection, and morphological operations for mask refinement. The output preserves the alpha channel to
indicate valid infill regions, enabling further processing or visualization.
"""

import cv2
import numpy as np
import os
import json

folder_img1 = r"C:\Users\Andrea\Desktop\Data\Stitched Images"
output_base_folder = r"C:\Users\Andrea\Desktop\Data\Second Version\Final Cut"

batch_jobs = [
    (r"C:\Users\Andrea\Desktop\Data\Second Version\Netfabb Slices_annotated", "005.png", ),
]

def get_suffix(filename):
    name, _ = os.path.splitext(filename)
    return name[-3:]

def match_images(folder1, folder2):
    imgs1 = {get_suffix(f): os.path.join(folder1, f) for f in os.listdir(folder1) if f.lower().endswith('.png')}
    imgs2 = {get_suffix(f): os.path.join(folder2, f) for f in os.listdir(folder2) if f.lower().endswith('.png')}
    matches = [(imgs1[k], imgs2[k]) for k in imgs1.keys() & imgs2.keys()]
    return matches

def scale_to_screen(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale

def get_saved_position(img1_path, img2_path):
    pos_x, pos_y = 50, 50
    scale_img2 = 1.0
    scale_img1 = 1.0
    drag = False
    ix, iy = -1, -1

    img1_orig = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_orig = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1_orig is None or img2_orig is None:
        print("Error loading images.")
        return None

    screen_w, screen_h = 1920, 1080
    max_w_img1 = int(screen_w * 0.7)
    max_h_img1 = int(screen_h * 0.7)
    img1, scale1 = scale_to_screen(img1_orig, max_w_img1, max_h_img1)
    img2, scale2 = scale_to_screen(img2_orig, screen_w, screen_h)  # <-- ora prendiamo scale2

    def mouse_events(event, x, y, flags, param):
        nonlocal pos_x, pos_y, drag, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            h2, w2 = int(img2.shape[0] * scale_img2), int(img2.shape[1] * scale_img2)
            if pos_x <= x <= pos_x + w2 and pos_y <= y <= pos_y + h2:
                drag = True
                ix, iy = x - pos_x, y - pos_y
        elif event == cv2.EVENT_LBUTTONUP:
            drag = False
        elif event == cv2.EVENT_MOUSEMOVE and drag:
            pos_x = x - ix
            pos_y = y - iy

    cv2.namedWindow("Position and Scale")
    cv2.setMouseCallback("Position and Scale", mouse_events)
    print(f"Manually align: {os.path.basename(img2_path)} on {os.path.basename(img1_path)}")
    print("Use the mouse to move IMG2 over IMG1.\n+/- to zoom IMG2 | z/x to zoom IMG1\nPress ENTER to confirm.")

    while True:
        # Effective scale of img1 relative to the original
        scala_effettiva_img1 = scale1 * scale_img1

        h1o, w1o = img1.shape[:2]
        w1 = int(w1o * scale_img1)
        h1 = int(h1o * scale_img1)
        img1_zoomed = cv2.resize(img1, (w1, h1), interpolation=cv2.INTER_AREA)
        vis = img1_zoomed.copy()

        h2o, w2o = img2.shape[:2]
        w2 = int(w2o * scale_img2)
        h2 = int(h2o * scale_img2)
        img2_resized = cv2.resize(img2, (w2, h2), interpolation=cv2.INTER_AREA)

        # Masks as before
        hsv_img2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_contour = cv2.inRange(hsv_img2, lower_green, upper_green)

        white_lower = np.array([240, 240, 240])
        white_upper = np.array([255, 255, 255])
        mask_white = cv2.inRange(img2_resized, white_lower, white_upper)
        mask_infill = cv2.bitwise_not(cv2.bitwise_or(mask_contour, mask_white))

        alpha_mask = np.zeros((h2, w2), dtype=np.float32)
        alpha_mask[mask_contour > 0] = 1.0
        alpha_mask[mask_infill > 0] = 0.4

        for c in range(3):
            y1, y2 = max(0, pos_y), min(pos_y + h2, vis.shape[0])
            x1, x2 = max(0, pos_x), min(pos_x + w2, vis.shape[1])
            if y1 >= y2 or x1 >= x2:
                continue
            roi_vis = vis[y1:y2, x1:x2, c]
            roi_img2 = img2_resized[(y1 - pos_y):(y2 - pos_y), (x1 - pos_x):(x2 - pos_x), c]
            roi_alpha = alpha_mask[(y1 - pos_y):(y2 - pos_y), (x1 - pos_x):(x2 - pos_x)]
            vis[y1:y2, x1:x2, c] = (roi_alpha * roi_img2 + (1 - roi_alpha) * roi_vis).astype(np.uint8)

        cv2.imshow("Position and Scale", vis)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            print("Manual exit.")
            exit()
        elif key == 13:
            cv2.destroyAllWindows()
            # Also return scale2 (initial scale of img2)
            print(f"DEBUG scales: scale1={scale1:.4f}, scale2={scale2:.4f}, scale_img1={scale_img1:.4f}, scale_img2={scale_img2:.4f}")
            return (pos_x, pos_y, scale_img2, scala_effettiva_img1, scale2)
        elif key == ord('+') or key == ord('='):
            scale_img2 = min(scale_img2 + 0.01, 3.0)
        elif key == ord('-') or key == ord('_'):
            scale_img2 = max(scale_img2 - 0.01, 0.1)
        elif key == ord('z'):
            scale_img1 = min(scale_img1 + 0.05, 3.0)
        elif key == ord('x'):
            scale_img1 = max(scale_img1 - 0.05, 0.1)

def process_pair(img1_path, img2_path, output_path, saved_pos, 
                 json_path=r"C:\Users\Andrea\Desktop\Data\Second Version\geometrie.json"):
    # Unpack parameters
    pos_x, pos_y, scale_img2, scala_effettiva_img1, scale2 = saved_pos
    img1_orig = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_orig = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1_orig is None or img2_orig is None:
        print(f"Loading error: {img1_path} or {img2_path}")
        return

    # Effective scale of img2 relative to its original
    scala_effettiva_img2 = scale2 * scale_img2

    # Original coordinates
    x_orig = int(round(pos_x / scala_effettiva_img1))
    y_orig = int(round(pos_y / scala_effettiva_img1))
    w_orig = int(round((img2_orig.shape[1] * scala_effettiva_img2) / scala_effettiva_img1))
    h_orig = int(round((img2_orig.shape[0] * scala_effettiva_img2) / scala_effettiva_img1))

    # Safe limits
    x_orig = max(0, min(x_orig, img1_orig.shape[1] - 1))
    y_orig = max(0, min(y_orig, img1_orig.shape[0] - 1))
    w_orig = max(1, min(w_orig, img1_orig.shape[1] - x_orig))
    h_orig = max(1, min(h_orig, img1_orig.shape[0] - y_orig))

    # Crop from img1
    crop = img1_orig[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig].copy()
    img2_resized = cv2.resize(img2_orig, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

    # Load JSON
    frames = []
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            frames = json.load(f)

    # Process each sub-crop separately
    for frame in frames:
        nome = frame['nome']
        x, y, w, h = frame['bbox']

        # Scale bbox coordinates from img2 -> crop
        scale_x = w_orig / img2_orig.shape[1]
        scale_y = h_orig / img2_orig.shape[0]
        x_crop = int(round(x * scale_x))
        y_crop = int(round(y * scale_y))
        w_crop = int(round(w * scale_x))
        h_crop = int(round(h * scale_y))

        x_crop = max(0, min(x_crop, crop.shape[1]-1))
        y_crop = max(0, min(y_crop, crop.shape[0]-1))
        w_crop = max(1, min(w_crop, crop.shape[1]-x_crop))
        h_crop = max(1, min(h_crop, crop.shape[0]-y_crop))

        sub_crop = crop[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        sub_img2 = img2_resized[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]

        # Calculate mask ONLY for this bbox
        hsv_sub = cv2.cvtColor(sub_img2, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_contour = cv2.inRange(hsv_sub, lower_green, upper_green)

        white_lower = np.array([240, 240, 240])
        white_upper = np.array([255, 255, 255])
        mask_white = cv2.inRange(sub_img2, white_lower, white_upper)
        mask_infill = cv2.bitwise_not(cv2.bitwise_or(mask_contour, mask_white))

        kernel = np.ones((3, 3), np.uint8)
        mask_contour = cv2.morphologyEx(mask_contour, cv2.MORPH_CLOSE, kernel)
        mask_infill = cv2.morphologyEx(mask_infill, cv2.MORPH_OPEN, kernel)

        if cv2.countNonZero(mask_infill) == 0:
            print(f"[{nome}] No infill found. Skipping.")
            continue

        # Create RGBA output only for this bbox
        output = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
        output[:, :, :3] = sub_crop
        output[:, :, 3] = 0
        output[:, :, 3][mask_infill > 0] = 255
        output[:, :, :3][mask_infill == 0] = 0

        # Save file
        numero = os.path.splitext(os.path.basename(img2_path))[0]
        nome_file = os.path.join(
            os.path.dirname(output_path),
            f"{nome}_{numero}.png"
        )
        cv2.imwrite(nome_file, output)
        print(f"Sub-crop saved: {nome_file}")

def main():
    saved_positions = []
    for folder, ref_img_name in batch_jobs:
        ref_img_path = os.path.join(folder, ref_img_name)
        print(f"Loading images from: {folder_img1} and {folder}")
        pairs = match_images(folder_img1, folder)
        # Search for the pair containing ref_img_name
        target_pair = None
        for p in pairs:
            if os.path.basename(p[1]) == ref_img_name:
                target_pair = p
                break
        if target_pair is None:
            print(f"Reference {ref_img_name} not found among images in {folder}")
            saved_positions.append(None)
            continue
        pos = get_saved_position(target_pair[0], target_pair[1])
        saved_positions.append(pos)

    print("\n--- Batch processing ---")
    for i, (folder, ref_img_name) in enumerate(batch_jobs):
        pos = saved_positions[i]
        if pos is None:
            print(f"Skipping folder {folder} due to missing positions.")
            continue
        pairs = match_images(folder_img1, folder)
        output_folder = output_base_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for img1_path, img2_path in pairs:
            suffix = get_suffix(img2_path)
            output_path = os.path.join(output_folder, f"{os.path.basename(folder)}_{suffix}_Processed.png")
            process_pair(img1_path, img2_path, output_path, pos)

if __name__ == "__main__":
    main()

