import os
import cv2
import numpy as np

from .core import process_dataset

# ---------------------------------------------------------
# Color Jitter (Brightness / Contrast / Saturation / Hue)
# ---------------------------------------------------------
def colorJitter(img,factor=[0.2, 0.2, 0.2]):
    #brightness=0.2
    #contrast=0.2
    #saturation=0.2
    brightness = factor[0]
    contrast = factor[1]
    saturation = factor[2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[...,2] *= (1 + np.random.uniform(-brightness, brightness))
    hsv[...,1] *= (1 + np.random.uniform(-saturation, saturation))
    hsv[...,0] += np.random.uniform(-10, 10)  # hue shift

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# ---------------------------------------------------------
#  BOXES - no change 
# ---------------------------------------------------------
def colorJitter_boxes(boxes):
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
def colorJitter_main (root_dir, output_dir, debug=False, verbose=True, factor=[0.2,0.2,0.2]):
    if verbose: 
        print(f'>> convert grayscale for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir,
                    output_dir,
                    colorJitter,  # func_img argument
                    colorJitter_boxes, # func_label argument
                    debug,
                    verbose,
                    factor)
    
    if verbose: 
        print(f'>> adjContrast completed ')