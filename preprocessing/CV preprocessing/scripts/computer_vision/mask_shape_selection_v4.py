"""
This script implements an interactive geometry selection and cropping system for images. The system allows users to
select geometries by clicking on regions, which triggers a flood fill algorithm to identify connected areas. Selected
geometries are named by the user and saved with their bounding box coordinates. The script then applies these selections
to all images in a specified folder, cropping each geometry region and saving the cropped images with descriptive names
that include the geometry name and the original image suffix. The implementation includes image resizing for display,
coordinate scaling between display and original image sizes, and batch processing capabilities for cropping entire
image folders.
"""

import cv2
import numpy as np
import os

# ========== SETUP ==========

img_path = r"C:\Users\Andrea\Desktop\Data\Netfabb Slices\001.png"
folder_path = r"C:\Users\Andrea\Desktop\Data\Netfabb Slices"

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

                geometrie_finali.append({'nome': nome, 'bbox': [x1, y1, w, h]})

            geometrie_correnti = []
        else:
            print("No geometry selected")

cv2.destroyAllWindows()

# ========== CREATE OUTPUT FOLDER FOR CROPS ==========


output_folder = folder_path + "_cropped"
os.makedirs(output_folder, exist_ok=True)

# ========== APPLY CROP TO ALL IMAGES ==========

for file in os.listdir(folder_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_file = os.path.join(folder_path, file)
        img = cv2.imread(img_file)
        if img is None:
            print(f"Error reading {file}")
            continue

        for g in geometrie_finali:
            x1, y1, w, h = g['bbox']
            nome = g['nome']

            # Assicurati che il ritaglio non superi i bordi immagine
            x2 = min(x1 + w, img.shape[1])
            y2 = min(y1 + h, img.shape[0])

            # Create a copy of the image to draw the box without modifying the original
            img_copy = img.copy()
            # Draw the green rectangle on the copy (thickness 2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ritaglio = img_copy[y1:y2, x1:x2]

            # Handle filename based on length
            save_suffix = file[-7:-4] if len(file) >= 7 else file.split('.')[-2]
            save_name = f"{nome}_{save_suffix}.png"

            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, ritaglio)
            print(f"Saved crop with border: {save_name}")

print(f"Crops completed. Crops saved to: {output_folder}")
