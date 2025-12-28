"""
This script implements an interactive geometry selection and annotation system for images, specifically designed for
processing Netfabb slice images. The system allows users to select geometries by clicking on regions, which triggers
a flood fill algorithm to identify connected areas. Selected geometries are named by the user and saved with their
bounding box coordinates in a JSON file. The script can then apply these annotations to all images in a specified
folder, drawing bounding boxes and labels on each image. The implementation includes image resizing for display,
coordinate scaling between display and original image sizes, and batch processing capabilities for annotating entire
image folders.
"""

import cv2
import numpy as np
import json
import os

# ========== SETUP ==========

img_path = r"C:\Users\Andrea\Desktop\Resized_Images\001.png"
folder_path = r"C:\Users\Andrea\Desktop\Resized_Images"
json_path = r"C:\Users\Andrea\Desktop\geometrie.json"
img_originale = cv2.imread(img_path)

# Resizing for display
max_width = 1200
scale = max_width / img_originale.shape[1]
width = int(img_originale.shape[1] * scale)
height = int(img_originale.shape[0] * scale)
img_display = cv2.resize(img_originale, (width, height))

# Clones for drawing
img_vista = img_display.copy()
img_lavoro = img_originale.copy()

# Variables
geometrie_correnti = []
geometrie_finali = []

def mouse_callback(event, x_vis, y_vis, flags, param):
    global img_vista, img_lavoro, geometrie_correnti

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert coordinates from resized to original
        x_orig = int(x_vis / scale)
        y_orig = int(y_vis / scale)

        # Flood fill on the original image
        mask = np.zeros((img_lavoro.shape[0]+2, img_lavoro.shape[1]+2), np.uint8)
        temp_img = img_lavoro.copy()
        retval, temp_img, mask, rect = cv2.floodFill(
            temp_img, mask, (x_orig, y_orig), (0, 255, 0),
            (5,)*3, (5,)*3, flags=4
        )

        if retval > 0:
            img_lavoro = temp_img
            geometrie_correnti.append({'rect': rect})
            print(f"Geometry selected at ({x_orig}, {y_orig})")
        else:
            print("Invalid click (white background?)")

cv2.namedWindow('Select geometries')
cv2.setMouseCallback('Select geometries', mouse_callback)

print("INSTRUCTIONS:")
print("- Click on geometries (in the resized window)")
print("- Press 'n' to assign a common name")
print("- Press 'q' to finish")

# ========== INTERACTION LOOP ==========

while True:
    cv2.imshow('Select geometries', img_vista)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        if geometrie_correnti:
            nome = input("Name to assign to these geometries: ")
            for g in geometrie_correnti:
                x1, y1, w, h = g['rect']
                cx = x1 + w // 2
                cy = y1 + h // 2

                # Draw on displayed image (so scale the coordinates)
                x1_s, y1_s = int(x1 * scale), int(y1 * scale)
                x2_s, y2_s = int((x1 + w) * scale), int((y1 + h) * scale)
                cx_s, cy_s = int(cx * scale), int(cy * scale)

                cv2.rectangle(img_vista, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
                cv2.putText(img_vista, nome, (cx_s - 30, cy_s), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Save data in the original version
                geometrie_finali.append({'nome': nome, 'bbox': [x1, y1, w, h]})

            geometrie_correnti = []
        else:
            print("No geometry selected")

cv2.destroyAllWindows()

# ========== SAVING ==========
with open(json_path, 'w') as f:
    json.dump(geometrie_finali, f, indent=4)
print(f"Geometries saved to {json_path}")

# ========== APPLY TO FOLDER ==========
output_folder = folder_path + "_annotated"
os.makedirs(output_folder, exist_ok=True)

with open(json_path, 'r') as f:
    geometrie = json.load(f)

for file in os.listdir(folder_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_file = os.path.join(folder_path, file)
        img = cv2.imread(img_file)
        if img is None:
            print(f"Error reading {file}")
            continue

        img_annotated = img.copy()

        for g in geometrie:
            x1, y1, w, h = g['bbox']
            nome = g['nome']
            cx = x1 + w // 2
            cy = y1 + h // 2

            cv2.rectangle(img_annotated, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        out_path = os.path.join(output_folder, file)
        cv2.imwrite(out_path, img_annotated)
        print(f"→ Annotated {file}")

print(f"Annotations completed. Files saved to: {output_folder}")

