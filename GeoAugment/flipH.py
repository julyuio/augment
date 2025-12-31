import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
# HORIZONTAL FLIP IMAGE
# ---------------------------------------------------------
def flipH(img,factor=0):
    flipped = cv2.flip(img, 1)  # 1 = horizontal flip
    return flipped


# ---------------------------------------------------------
# HORIZONTAL FLIP YOLO BOXES
# ---------------------------------------------------------
def flip_yolo_boxes_horizontal(boxes):
    flipped_boxes = []

    for cls, xc, yc, bw, bh in boxes:
        # Horizontal flip: x_center becomes (1 - x_center)
        new_xc = 1.0 - xc
        new_yc = yc
        new_bw = bw
        new_bh = bh

        flipped_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])

    return flipped_boxes


def flipH_main (root_dir,output_dir, debug=False, verbose=True):
    if verbose:
        print(f'>> flipping along horizontal : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir = root_dir,
                     output_dir = output_dir,
                     func_img  =flipH, 
                     func_label = flip_yolo_boxes_horizontal, 
                     debug = debug, 
                     verbose = verbose)
    
    if verbose: 
        print(f'>> flipH completed ')



