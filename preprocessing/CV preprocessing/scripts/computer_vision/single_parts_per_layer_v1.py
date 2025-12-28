"""
This script implements an interactive cropping system that allows users to define multiple crop regions on a reference
image and then applies these crops to all images in a folder. The system first prompts the user to select a reference
folder and specify the number of crops to define. For each crop, the user interactively selects two opposite vertices
of a rectangle using mouse clicks, with support for zoom and pan operations. The crops are named by the user and their
coordinates are saved in a JSON file. The script then applies these crops to all images in the folder, extracting the
corresponding regions and saving them with descriptive names. The implementation includes screen-adaptive image resizing,
interactive point selection with zoom/pan capabilities, and batch processing of images with consistent crop parameters.
Note: Input images must follow a specific naming convention, ending with a number followed by an underscore and another
expression (e.g., "name1_L1").
"""

import cv2
import os
import re
import json
import numpy as np
from tqdm import tqdm

# Path
input_folder = r"C:\Users\Andrea\Desktop\Data\Netfabb Slices Edges"

# ====================== Screen ======================
def get_screen_size(fallback=(1280, 720)):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return fallback

def resize_to_screen(img, fill_ratio=0.9, no_upscale=True):
    h, w = img.shape[:2]
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * fill_ratio)
    max_h = int(screen_h * fill_ratio)
    scale = min(max_w / w, max_h / h)
    if no_upscale:
        scale = min(scale, 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    else:
        resized = img.copy()
    return resized, scale

# ====================== Utility ======================
def find_lowest_numbered_png(folder_path):
    pattern = re.compile(r"(\d+)\.png$", re.IGNORECASE)
    candidates = []
    for f in os.listdir(folder_path):
        match = pattern.search(f)
        if match:
            num = int(match.group(1))
            candidates.append((num, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return os.path.join(folder_path, candidates[0][1])

def extract_suffix_from_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    return name.split("_")[-1] if "_" in name else name

# ====================== Points Selection ======================
def select_two_points(img):
    display, scale = resize_to_screen(img)
    points = []
    zoom, zoom_step, max_zoom, min_zoom = 1.0, 0.1, 5.0, 0.5
    base_w, base_h = display.shape[1], display.shape[0]
    offset_x, offset_y = 0, 0
    dragging_pan = False
    pan_start, offset_start = (0, 0), (0, 0)

    print("""
INSTRUCTIONS:
- Left-click to select 2 opposite vertices of the rectangle to crop.
- Use the MOUSE WHEEL to ZOOM in/out.
- Press and HOLD the MOUSE WHEEL and move the mouse to PAN the image.
- Press ENTER when you have selected 2 points to confirm.
- Press ESC to exit without saving.
Press any key to continue...""")
    cv2.waitKey(0)

    def mouse_cb(event, x, y, flags, param):
        nonlocal zoom, offset_x, offset_y, dragging_pan, pan_start, offset_start, points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            px = int((x - offset_x) / zoom)
            py = int((y - offset_y) / zoom)
            if 0 <= px < base_w and 0 <= py < base_h:
                points.append((px, py))
                print(f"Point selected: {px}, {py}")
        elif event == cv2.EVENT_MBUTTONDOWN:
            dragging_pan, pan_start, offset_start = True, (x, y), (offset_x, offset_y)
        elif event == cv2.EVENT_MBUTTONUP:
            dragging_pan = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging_pan:
            dx, dy = x - pan_start[0], y - pan_start[1]
            offset_x = max(min(offset_start[0] + dx, 0), -max(0, int(base_w * zoom) - base_w))
            offset_y = max(min(offset_start[1] + dy, 0), -max(0, int(base_h * zoom) - base_h))
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = zoom
            zoom = min(max_zoom, zoom + zoom_step) if flags > 0 else max(min_zoom, zoom - zoom_step)
            mx, my = x, y
            offset_x = int((offset_x - mx) * (zoom / old_zoom) + mx)
            offset_y = int((offset_y - my) * (zoom / old_zoom) + my)
            offset_x = max(min(offset_x, 0), -max(0, int(base_w * zoom) - base_w))
            offset_y = max(min(offset_y, 0), -max(0, int(base_h * zoom) - base_h))

    cv2.namedWindow("Select two opposite vertices", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select two opposite vertices", display.shape[1], display.shape[0])
    cv2.setMouseCallback("Select two opposite vertices", mouse_cb)

    while True:
        zoomed = cv2.resize(display, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        canvas = zoomed[-offset_y: -offset_y + base_h, -offset_x: -offset_x + base_w].copy()

        for p in points:
            px, py = int(p[0] * zoom) + offset_x, int(p[1] * zoom) + offset_y
            cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        if len(points) == 2:
            pt1 = (int(points[0][0] * zoom) + offset_x, int(points[0][1] * zoom) + offset_y)
            pt2 = (int(points[1][0] * zoom) + offset_x, int(points[1][1] * zoom) + offset_y)
            cv2.rectangle(canvas, pt1, pt2, (0, 0, 255), 2)

        cv2.imshow("Select two opposite vertices", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key in (13, 10) and len(points) == 2:
            cv2.destroyAllWindows()
            return [(int(p[0] / scale), int(p[1] / scale)) for p in points]

# ====================== MAIN ======================
def main():
    if not os.path.isdir(input_folder):
        print("Invalid path or not a directory.")
        return

    parent_dir = os.path.dirname(os.path.abspath(input_folder))
    save_folder = os.path.join(parent_dir, "Netfabb Slices Edges Cropped")
    os.makedirs(save_folder, exist_ok=True)

    img_path = find_lowest_numbered_png(input_folder)
    if img_path is None:
        print("No .png image found in the provided folder.")
        return
    print(f"Image chosen for cropping selection: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("Image not found or cannot be loaded.")
        return

    try:
        n_crops = int(input("How many crops do you want to select? "))
    except ValueError:
        print("Invalid number.")
        return

    crops = []
    for i in range(1, n_crops + 1):
        print(f"\nSelect crop {i}:")
        pts = select_two_points(img)
        if pts is None:
            print("Early exit during crop selection.")
            return
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        name = input(f"Crop name {i} (without extension): ").strip() or f"crop_{i}"
        crops.append({"name": name, "coordinates": [x1, y1, x2, y2]})

    # Save JSON file
    json_path = os.path.join(parent_dir, "netfabb_slices_crop.json")
    with open(json_path, "w") as jf:
        json.dump(crops, jf, indent=4)
    print(f"\nCrop data saved to {json_path}")

    # Apply crops to all images
    pattern = re.compile(r"(\d+)\.png$", re.IGNORECASE)
    all_images = [f for f in os.listdir(input_folder) if pattern.search(f)]
    print(f"\nStarting cropping and saving {len(all_images)} images...")

    for img_file in tqdm(all_images, desc="Processing images"):
        img_path_full = os.path.join(input_folder, img_file)
        im = cv2.imread(img_path_full)
        if im is None:
            print(f"Image {img_file} cannot be loaded, skipping.")
            continue

        suffix = extract_suffix_from_filename(img_file)
        for crop in crops:
            x1, y1, x2, y2 = crop["coordinates"]
            crop_img = im[y1:y2, x1:x2]
            out_name = f"{crop['name']}_{suffix}.png"
            out_path = os.path.join(save_folder, out_name)
            cv2.imwrite(out_path, crop_img)

    print(f"All cropped images saved in: {save_folder}")

if __name__ == "__main__":
    main()
