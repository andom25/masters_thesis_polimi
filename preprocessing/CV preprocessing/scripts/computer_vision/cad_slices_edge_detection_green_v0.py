"""
This script implements an edge detection algorithm for identifying contours in images characterized by white backgrounds
and colorful shapes. The algorithm processes images from an input folder and extracts the initial contour from the first
image, which is then applied as a green border overlay to all subsequent images. The implementation uses OpenCV's contour
detection with morphological operations to precisely identify boundaries between white and non-white regions.
It is critical that the first image in the folder corresponds to the very first layer, ensuring that the identified
contour represents the most external boundary across all images in the sequence.
"""

import cv2
import numpy as np
import os
import re

# Paths
input_folder = r"C:\Users\spierings\Desktop\Coldani_Domini_BJ2\netfabb_slices"
output_folder = r"C:\Users\spierings\Desktop\netfabb_slices_edge"

# Colors
GREEN = (0, 0, 0)

def find_precise_contours(img_rgb):
    """
    Returns contours between white and non-white areas, with better definition.
    """
    mask = cv2.inRange(img_rgb, np.array([255, 255, 255]), np.array([255, 255, 255]))
    mask_inv = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask_inv_dilated = cv2.dilate(mask_inv, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask_inv_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def sort_files_by_number(file_list):
    def extract_number(name):
        match = re.search(r"(\d+)\.png$", name)
        return int(match.group(1)) if match else -1
    return sorted(file_list, key=extract_number)

def process_images(input_folder, output_folder, green_border_thickness=2):
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    png_files = sort_files_by_number(png_files)

    if not png_files:
        print("No images found in the folder.")
        return

    initial_contours = None

    for i, file_name in enumerate(png_files):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_result = img_rgb.copy()

        contours, hierarchy = find_precise_contours(img_rgb)

        if i == 0:
            initial_contours = contours
            cv2.drawContours(img_result, initial_contours, -1, GREEN, green_border_thickness)
        else:
            cv2.drawContours(img_result, initial_contours, -1, GREEN, green_border_thickness)

        img_final = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, img_final)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    process_images(input_folder, output_folder)

