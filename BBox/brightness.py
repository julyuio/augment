import os
import cv2
import numpy as np

# ---------------------------------------------------------
# ADJUST BRIGHTNESS
# ---------------------------------------------------------
def adjBrightness(img, delta=40):
    """
    Adjust brightness by adding a constant value.
    delta > 0 → brighter
    delta < 0 → darker
    """
    img = img.astype(np.int16)
    img = img + delta
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------------------------------------------------------
#  BOXES - no change 
# ---------------------------------------------------------
def adjBrightness_boxes(boxes):
    new_boxes = []
    for cls, xc, yc, bw, bh in boxes:
        new_xc = xc
        new_yc = yc
        new_bw = bw
        new_bh = bh
        new_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])
    return new_boxes



def adjBrightness_main (root_dir, output_dir, debug=False, verbose=True, factor=20):
    if verbose: 
        print(f'>> adjusting brightness for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir,
                    output_dir,
                    adjBrightness,  # func_img argument
                    adjContrast_boxes, # func_label argument
                    debug,
                    verbose,
                    factor)
    
    if verbose: 
        print(f'>> adjBrightness completed ')


