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
# Main entry from __init__ 
# ---------------------------------------------------------
def flipV_main (root_dir, output_dir, debug=False, verbose=True):
    if verbose: 
        print(f'>> flipping along vertical : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir, output_dir, flipV, flip_yolo_boxes_vertical, debug, verbose)
    
    if verbose: 
        print(f'>> flipV completed ')

