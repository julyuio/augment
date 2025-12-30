import os
import cv2
import numpy as np

# ---------------------------------------------------------
# ADJUST CONTRAST
# ---------------------------------------------------------
def adjContrast(img, factor=1.5):
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
# # PROCESS DATASET
# # ---------------------------------------------------------
# def process_dataset(root_dir, output_dir, contrast_factor=1.5):
#     img_dir = os.path.join(root_dir, "images")
#     lbl_dir = os.path.join(root_dir, "labels")

#     out_img_dir = os.path.join(output_dir, "images")
#     out_lbl_dir = os.path.join(output_dir, "labels")

#     debug_img_dir = os.path.join(output_dir + "_debug", "images")
#     debug_lbl_dir = os.path.join(output_dir + "_debug", "labels")

#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_lbl_dir, exist_ok=True)
#     os.makedirs(debug_img_dir, exist_ok=True)
#     os.makedirs(debug_lbl_dir, exist_ok=True)

#     for fname in os.listdir(img_dir):
#         if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
#             continue

#         img_path = os.path.join(img_dir, fname)
#         txt_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

#         if not os.path.exists(txt_path):
#             print(f"Skipping {fname}: no label file")
#             continue

#         # Load image
#         img = cv2.imread(img_path)

#         # Load YOLO labels
#         boxes = []
#         with open(txt_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 cls = int(parts[0])
#                 xc, yc, bw, bh = map(float, parts[1:])
#                 boxes.append([cls, xc, yc, bw, bh])

#         # Apply contrast adjustment
#         contrast_img = adjust_contrast(img, factor=contrast_factor)

#         # Save contrast-adjusted image
#         out_img_path = os.path.join(out_img_dir, fname)
#         cv2.imwrite(out_img_path, contrast_img)

#         # Save labels (unchanged)
#         out_txt_path = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")
#         with open(out_txt_path, "w") as f:
#             for cls, xc, yc, bw, bh in boxes:
#                 f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

#         # Debug image with drawn boxes
#         debug_img = draw_boxes(contrast_img, boxes)
#         cv2.imwrite(os.path.join(debug_img_dir, fname), debug_img)

#         # Copy labels to debug folder
#         with open(os.path.join(debug_lbl_dir, os.path.splitext(fname)[0] + ".txt"), "w") as f:
#             for cls, xc, yc, bw, bh in boxes:
#                 f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

#         print(f"Processed {fname}")


# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def adjContrast_main (root_dir, output_dir, debug=False, verbose=True):
    if verbose: 
        print(f'>> flipping along vertical : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect)
    process_dataset(root_dir, output_dir, adjContrast, None, debug, verbose)
    
    if verbose: 
        print(f'>> flipV completed ')
