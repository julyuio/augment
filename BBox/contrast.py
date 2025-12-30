import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
# ADJUST CONTRAST
# ---------------------------------------------------------
def adjContrast(img, factor):
    """
    Adjust contrast by scaling pixel values around the midpoint.
    factor > 1.0 → higher contrast
    factor < 1.0 → lower contrast
    """
    img = img.astype(np.float32)
    img = (img - 127.5) * factor + 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------------------------------------------------------
#  BOXES - no change 
# ---------------------------------------------------------
def adjContrast_boxes(boxes):
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
def adjContrast_main (root_dir, output_dir, debug=False, verbose=True, factor=1.5):
    if verbose: 
        print(f'>> adjusting contrast for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir,
                    output_dir,
                    adjContrast,  # func_img argument
                    adjContrast_boxes, # func_label argument
                    debug,
                    verbose,
                    factor)
    
    if verbose: 
        print(f'>> adjContrast completed ')
