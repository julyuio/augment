import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
# VERTICAL FLIP IMAGE
# ---------------------------------------------------------
def flipV(img):
    flipped = cv2.flip(img, 0)  # 0 = vertical flip
    return flipped


# ---------------------------------------------------------
# VERTICAL FLIP YOLO BOXES
# ---------------------------------------------------------
def flip_yolo_boxes_vertical(boxes):
    flipped_boxes = []

    for cls, xc, yc, bw, bh in boxes:
        # Vertical flip: y_center becomes (1 - y_center)
        new_xc = xc
        new_yc = 1.0 - yc
        new_bw = bw
        new_bh = bh

        flipped_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])

    return flipped_boxes


# ---------------------------------------------------------
# DRAW DEBUG BOXES
# ---------------------------------------------------------
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    debug_img = img.copy()

    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(debug_img, str(cls), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return debug_img


# ---------------------------------------------------------
# Main entry function
# ---------------------------------------------------------
def flipV_main (root_dir, output_dir, debug=False, verbose=True):
    if verbose: 
        print(f'>> flipping along vertical : {root_dir}')
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir, output_dir, flipV, flip_yolo_boxes_vertical, debug, verbose)
    if verbose: 
        print(f'>> flipV completed ')


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
# if __name__ == "__main__":
#     process_dataset(
#         root_dir="train",
#         output_dir="train_flipped_vertical"
#     )

