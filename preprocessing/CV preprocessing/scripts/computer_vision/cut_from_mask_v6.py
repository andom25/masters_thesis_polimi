"""
This script implements an interactive image alignment and cropping system for processing pairs of images from two folders.
The system enables manual alignment of a second image (Netfabb slice) over a first image (buildplate) through user interaction,
allowing position and scale adjustments via mouse and keyboard controls. After alignment, the script extracts regions based on
mask detection, identifying infill areas (non-contour, non-white regions) using HSV color space and morphological operations.
The cropped regions are saved as RGBA images with transparency, where the alpha channel indicates valid infill regions.
The implementation supports batch processing of multiple image pairs with consistent alignment parameters.
"""

import cv2
import numpy as np
import os

folder_img1 = r"C:\Users\Andrea\Desktop\Data\Stitched Images"
output_base_folder = r"C:\Users\Andrea\Desktop\Data\Final Images"

batch_jobs = [
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CC1", "CC1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CC2", "CC2_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CGA1", "CGA1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CGA2", "CGA2_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CGB1", "CGB1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\CGB2", "CGB2_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DC1", "DC1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DC2", "DC2_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DGA1", "DGA1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DGA2", "DGA2_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DGB1", "DGB1_200.png"),
    (r"C:\Users\Andrea\Desktop\Data\Netfabb Slices_cropped\DGB2", "DGB2_200.png")
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
    img2, _ = scale_to_screen(img2_orig, screen_w, screen_h)

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
        h1o, w1o = img1.shape[:2]
        w1 = int(w1o * scale_img1)
        h1 = int(h1o * scale_img1)
        img1_zoomed = cv2.resize(img1, (w1, h1), interpolation=cv2.INTER_AREA)
        vis = img1_zoomed.copy()

        h2o, w2o = img2.shape[:2]
        w2 = int(w2o * scale_img2)
        h2 = int(h2o * scale_img2)
        img2_resized = cv2.resize(img2, (w2, h2), interpolation=cv2.INTER_AREA)

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
            return (pos_x, pos_y, scale_img2, scale1 * scale_img1)
        elif key == ord('+') or key == ord('='):
            scale_img2 = min(scale_img2 + 0.01, 3.0)
        elif key == ord('-') or key == ord('_'):
            scale_img2 = max(scale_img2 - 0.01, 0.1)
        elif key == ord('z'):
            scale_img1 = min(scale_img1 + 0.05, 3.0)
        elif key == ord('x'):
            scale_img1 = max(scale_img1 - 0.05, 0.1)

def process_pair(img1_path, img2_path, output_path, saved_pos):
    pos_x, pos_y, scale_img2, scale_total = saved_pos
    img1_orig = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_orig = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1_orig is None or img2_orig is None:
        print(f"Loading error: {img1_path} or {img2_path}")
        return

    w2 = int(img2_orig.shape[1] * scale_img2)
    h2 = int(img2_orig.shape[0] * scale_img2)
    x_orig = int(pos_x / scale_total)
    y_orig = int(pos_y / scale_total)
    w_orig = int(w2 / scale_total)
    h_orig = int(h2 / scale_total)

    # Limits for cropping in img1_orig
    x_orig = max(0, min(x_orig, img1_orig.shape[1] - 1))
    y_orig = max(0, min(y_orig, img1_orig.shape[0] - 1))
    w_orig = min(w_orig, img1_orig.shape[1] - x_orig)
    h_orig = min(h_orig, img1_orig.shape[0] - y_orig)

    crop = img1_orig[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig].copy()
    img2_resized = cv2.resize(img2_orig, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

    hsv_img2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask_contour = cv2.inRange(hsv_img2, lower_green, upper_green)

    white_lower = np.array([240, 240, 240])
    white_upper = np.array([255, 255, 255])
    mask_white = cv2.inRange(img2_resized, white_lower, white_upper)

    # Infill: everything that is NOT contour nor white
    mask_infill = cv2.bitwise_not(cv2.bitwise_or(mask_contour, mask_white))

    kernel = np.ones((3, 3), np.uint8)
    mask_contour = cv2.morphologyEx(mask_contour, cv2.MORPH_CLOSE, kernel)
    mask_infill = cv2.morphologyEx(mask_infill, cv2.MORPH_OPEN, kernel)

    if cv2.countNonZero(mask_infill) == 0:
        print(f"No infill found in {os.path.basename(img2_path)}. Skipping.")
        return

    # Create output with alpha channel
    output = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

    # Copy the crop to the first 3 RGB channels where there is infill (mask_infill)
    for c in range(3):
        output[:, :, c] = crop[:, :, c]

    # Set alpha to 0 (transparent) everywhere
    output[:, :, 3] = 0

    # Where there is infill, alpha = 255 (opaque)
    output[:, :, 3][mask_infill > 0] = 255

    # And for safety, keep only the part corresponding to infill in RGB
    for c in range(3):
        output[:, :, c][mask_infill == 0] = 0

    # Save with transparency
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    cv2.imwrite(output_path, output)
    print(f"Output saved to {output_path}")


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
        output_folder = os.path.join(output_base_folder, os.path.basename(folder))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for img1_path, img2_path in pairs:
            suffix = get_suffix(img2_path)
            output_path = os.path.join(output_folder, f"{os.path.basename(folder)}_{suffix}.png")
            process_pair(img1_path, img2_path, output_path, pos)

if __name__ == "__main__":
    main()
