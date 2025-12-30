import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
#  reduce saturation by a factor.
# ---------------------------------------------------------
def desaturate(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= (1 - factor)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------
#  BOXES - no change 
# ---------------------------------------------------------
def desaturate_boxes(boxes):
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
def desaturate_main (root_dir, output_dir, debug=False, verbose=True, factor=1.5):
    if verbose: 
        print(f'>> desaturate for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir,
                    output_dir,
                    desaturate,  # func_img argument
                    desaturate_boxes, # func_label argument
                    debug,
                    verbose,
                    factor)
    
    if verbose: 
        print(f'>> desaturate completed ')

        