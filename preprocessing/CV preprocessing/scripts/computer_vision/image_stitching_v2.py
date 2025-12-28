"""
This script implements an interactive image stitching system that processes images in user-defined groups and combines
selected vertical regions from each image. The system allows users to manually select two vertical cut positions per image
through mouse clicks, defining the region to be extracted. Images are loaded from a specified folder, and the stitched
results are saved in a new folder while preserving the original images. The cut coordinates are saved in a JSON file for
reproducibility. The implementation includes image resizing for display, interactive cut selection with visual feedback,
and horizontal concatenation of selected regions to create the final stitched images.
"""

import os
import re
import cv2
import json
from tqdm import tqdm

# Insert here the path to the BRIGHT FIELD images folder
BF_FOLDER_PATH = r"C:\Users\Andrea\Desktop\temp"
# Set here the number of images to stitch together
GROUP_SIZE = 2

def get_images_by_number(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    images = {}
    for fname in files:
        m = re.search(r'(\d+)(?=\.[^.]+$)', fname)
        if m:
            images[int(m.group(1))] = fname
    return images

def get_nonoverlapping_groups(images_dict, group_size):
    nums = sorted(images_dict.keys())
    groups = []
    for i in range(0, len(nums) - group_size + 1, group_size):
        group_nums = nums[i:i+group_size]
        if len(group_nums) == group_size:
            groups.append(tuple(images_dict[n] for n in group_nums))
    return groups

def resize_image_keep_aspect(img, max_size=900):
    h, w = img.shape[:2]
    scale_w = max_size / w
    scale_h = max_size / h
    scale = min(scale_w, scale_h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def select_vertical_cuts_on_images(folder, group):
    cut_positions = []
    cut_coords_dict = {}

    for img_name in group:
        img_path = os.path.join(folder, img_name)
        img_full = cv2.imread(img_path)
        if img_full is None:
            print(f"⚠️ Cannot read image {img_name}")
            return None

        img_display = resize_image_keep_aspect(img_full, max_size=900)
        disp_h, disp_w = img_display.shape[:2]

        fixed_cuts = []
        cursor_x = 0
        done = False

        window_name = f"Select vertical cuts - {img_name} (Click twice to place cuts)"

        def draw(img, cursor_x):
            img_copy = img.copy()
            for x in fixed_cuts:
                cv2.line(img_copy, (x,0), (x,disp_h), (0,0,255), 1)
            if len(fixed_cuts) < 2:
                cv2.line(img_copy, (cursor_x,0), (cursor_x,disp_h), (0,0,255), 1)
            texts = [
                f"Image width: {disp_w} px",
                f"Cursor X: {cursor_x}",
            ]
            if len(fixed_cuts) > 0:
                texts.append(f"Cut 1: {fixed_cuts[0]}")
            if len(fixed_cuts) > 1:
                texts.append(f"Cut 2: {fixed_cuts[1]}")
            for i, t in enumerate(texts):
                cv2.putText(img_copy, t, (10, 20 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            return img_copy

        def mouse_cb(event, x, y, flags, param):
            nonlocal fixed_cuts, done, cursor_x
            if event == cv2.EVENT_MOUSEMOVE:
                cursor_x = x
            elif event == cv2.EVENT_LBUTTONDOWN:
                if len(fixed_cuts) < 2:
                    fixed_cuts.append(x)
                    fixed_cuts.sort()
                    if len(fixed_cuts) == 2:
                        done = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_cb)

        while True:
            img_to_show = draw(img_display, cursor_x)
            cv2.imshow(window_name, img_to_show)
            k = cv2.waitKey(20) & 0xFF
            if done:
                break
            if k == 27:
                cv2.destroyWindow(window_name)
                return None

        cv2.destroyWindow(window_name)

        scale_x = img_full.shape[1] / disp_w
        x1 = int(fixed_cuts[0] * scale_x)
        x2 = int(fixed_cuts[1] * scale_x)
        x1, x2 = sorted([x1, x2])

        cut_positions.append(slice(x1, x2))
        cut_coords_dict[img_name] = [x1, x2]

    output_json_path = os.path.join(os.path.dirname(folder), "cut_coordinates.json")
    try:
        with open(output_json_path, "w") as f:
            json.dump(cut_coords_dict, f, indent=2)
        print(f"\n📁 Saved cut positions to: {output_json_path}")
    except Exception as e:
        print(f"⚠️ Failed to save cut coordinates: {e}")

    return cut_positions

def stitch_images_by_user_groups(bf_dir, group_size):
    images_dict = get_images_by_number(bf_dir)

    while True:
        groups = get_nonoverlapping_groups(images_dict, group_size)
        total_images = len(images_dict)
        expected = group_size * len(groups)
        if expected != total_images:
            print(f"Number of images ({total_images}) not multiple of group size ({group_size}). Please change the GROUP_SIZE parameter.")
            return
        else:
            break

    print(f"\nGroups ({group_size} images each):")
    for i, g in enumerate(groups):
        print(f" Group {i+1}: {', '.join(g)}")
    input("Press Enter to start cut selection or Ctrl+C to cancel...")

    cuts_slices = select_vertical_cuts_on_images(bf_dir, groups[0])
    if cuts_slices is None:
        print("Cut selection cancelled.")
        return

    output_dir = os.path.join(os.path.dirname(bf_dir), 'Stitched Images')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Stitching images in groups of {group_size}...")

    for i, group_files in enumerate(tqdm(groups, desc="Stitching images")):
        parts = []
        for j, f in enumerate(group_files):
            img = cv2.imread(os.path.join(bf_dir, f))
            if img is None:
                print(f"Skipping group {i+1} due to error reading {f}")
                continue
            y_slice = slice(0, img.shape[0])
            x_slice = cuts_slices[j]
            part = img[y_slice, x_slice]
            parts.append(part)
        stitched = cv2.hconcat(parts)
        output_name = f'Layer_{i+1:03d}.png'
        cv2.imwrite(os.path.join(output_dir, output_name), stitched)

    print("Done! All stitched images saved.")

if __name__ == '__main__':
    bf_dir = BF_FOLDER_PATH
    if not os.path.isdir(bf_dir):
        print(f"Folder not found: {bf_dir}")
        exit(0)

    stitch_images_by_user_groups(bf_dir, GROUP_SIZE)







