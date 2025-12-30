import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
# CONVERT IMAGE TO GRAYSCALE (3-channel output)
# ---------------------------------------------------------
def convertGrayscale(img,factor=0):
    """
    Converts image to grayscale but returns 3-channel output
    so YOLO models can still train on it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_3ch


# ---------------------------------------------------------
#  BOXES - no change 
# ---------------------------------------------------------
def convertGrayscale_boxes(boxes):
    new_boxes = []
    for cls, xc, yc, bw, bh in boxes:
        new_xc = xc
        new_yc = yc
        new_bw = bw
        new_bh = bh
        new_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])
    return new_boxes


# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def convertGrayscale_main (root_dir, output_dir, debug=False, verbose=True, factor=1.5):
    if verbose: 
        print(f'>> convert grayscale for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir,
                    output_dir,
                    convertGrayscale,  # func_img argument
                    convertGrayscale_boxes, # func_label argument
                    debug,
                    verbose,
                    factor)
    
    if verbose: 
        print(f'>> adjContrast completed ')