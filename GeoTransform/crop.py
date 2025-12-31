import os
import cv2
import numpy as np


from .core import draw_boxes, copy_boxes

# ---------------------------------------------------------
# CROP A SINGLE OBJECT FROM AN IMAGE
# ---------------------------------------------------------
def crop_object(img, box, padding=0.05):
    """
    img: original image
    box: [cls, xc, yc, bw, bh] in YOLO normalized format
    padding: % of bbox size to add around crop
    """
    h, w = img.shape[:2]
    cls, xc, yc, bw, bh = box

    # Convert YOLO → pixel coords
    xc *= w; yc *= h
    bw *= w; bh *= h

    x1 = int(xc - bw/2)
    y1 = int(yc - bh/2)
    x2 = int(xc + bw/2)
    y2 = int(yc + bh/2)

    # Add padding

    padding = factor
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    # Crop image
    crop = img[y1:y2, x1:x2]

    # New YOLO box inside the crop (always centered)
    new_w = x2 - x1
    new_h = y2 - y1

    new_xc = (xc - x1) / new_w
    new_yc = (yc - y1) / new_h
    new_bw = bw / new_w
    new_bh = bh / new_h

    return crop, [cls, new_xc, new_yc, new_bw, new_bh]


# ---------------------------------------------------------
# DRAW DEBUG BOXES
# ---------------------------------------------------------
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    debug_img = img.copy()

    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(debug_img, str(cls), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return debug_img


# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------
def process_dataset_crop(root_dir, output_dir, padding=0.05):
    img_dir = os.path.join(root_dir, "images")
    lbl_dir = os.path.join(root_dir, "labels")

    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")

    debug_img_dir = os.path.join(output_dir + "_debug", "images")
    debug_lbl_dir = os.path.join(output_dir + "_debug", "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(debug_img_dir, exist_ok=True)
    os.makedirs(debug_lbl_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, fname)
        txt_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

        if not os.path.exists(txt_path):
            print(f"Skipping {fname}: no label file")
            continue

        img = cv2.imread(img_path)

        # Load YOLO labels
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                boxes.append([cls, xc, yc, bw, bh])

        # Create a crop for each object
        for idx, box in enumerate(boxes):
            crop_img, crop_box = crop_object(img, box, padding=padding)

            crop_name = f"{os.path.splitext(fname)[0]}_crop_{idx}.jpg"
            label_name = f"{os.path.splitext(fname)[0]}_crop_{idx}.txt"

            # Save crop
            cv2.imwrite(os.path.join(out_img_dir, crop_name), crop_img)

            # Save label
            with open(os.path.join(out_lbl_dir, label_name), "w") as f:
                cls, xc, yc, bw, bh = crop_box
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            # Debug image
            debug_img = draw_boxes(crop_img, [crop_box])
            cv2.imwrite(os.path.join(debug_img_dir, crop_name), debug_img)

            # Copy label to debug
            with open(os.path.join(debug_lbl_dir, label_name), "w") as f:
                cls, xc, yc, bw, bh = crop_box
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"Processed {fname}")


# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def crop_main (root_dir, output_dir, debug=False, verbose=True, factor=0.05):
    if verbose: 
        print(f'>> crop for : {root_dir}')
    
    # process dataset is main function in core.py that repeats for all other actions/tasks (flipV, flipH, brightness.... ect) except rotate
    process_dataset_crop(root_dir = root_dir ,
                    output_dir = output_dir ,
                    func_img = desaturate,  # func_img argument
                    func_label = copy_boxes, # func_label argument
                    debug = debug , 
                    verbose = verbose,
                    factor = factor) # 5% padding around each crop
    
    if verbose: 
        print(f'>> desaturate completed ')



# # ---------------------------------------------------------
# # RUN
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     process_dataset(
#         root_dir="train",
#         output_dir="train_crops",
#         padding=0.05  # 5% padding around each crop
#     )