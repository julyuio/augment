import os
import cv2
import numpy as np

# ---------------------------------------------------------
# ROTATE IMAGE WITHOUT CROPPING
# ---------------------------------------------------------
def rotate_img(img, factor=90):
    angle_cw = factor
    h, w = img.shape[:2]
    angle_ccw = -angle_cw  # OpenCV rotates CCW

    # Rotation matrix around center
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_ccw, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust matrix to translate image to center of new canvas
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate with expanded canvas
    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))
    return rotated, M, new_w, new_h


# ---------------------------------------------------------
# ROTATE YOLO BOXES FOR EXPANDED CANVAS
# ---------------------------------------------------------
def rotate_yolo_boxes_no_crop(boxes, M, orig_w, orig_h, new_w, new_h):
    new_boxes = []

    for cls, xc, yc, bw, bh in boxes:
        # Convert YOLO normalized → pixel coords
        xc *= orig_w; yc *= orig_h
        bw *= orig_w; bh *= orig_h

        # Original corners
        x1 = xc - bw / 2; y1 = yc - bh / 2
        x2 = xc + bw / 2; y2 = yc + bh / 2

        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])

        # Rotate corners
        rotated = (M @ corners.T).T

        # New bounding box
        x_min = rotated[:, 0].min()
        y_min = rotated[:, 1].min()
        x_max = rotated[:, 0].max()
        y_max = rotated[:, 1].max()

        # Clip to new canvas
        x_min = max(0, min(new_w, x_min))
        y_min = max(0, min(new_h, y_min))
        x_max = max(0, min(new_w, x_max))
        y_max = max(0, min(new_h, y_max))

        # Convert back to YOLO normalized
        new_xc = (x_min + x_max) / 2 / new_w
        new_yc = (y_min + y_max) / 2 / new_h
        new_bw = (x_max - x_min) / new_w
        new_bh = (y_max - y_min) / new_h

        new_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])

    return new_boxes


# ---------------------------------------------------------
# DRAW DEBUG BOXES
# ---------------------------------------------------------
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2, alpha=0.3):
    h, w = img.shape[:2]
    debug_img = img.copy()
    overlay = img.copy()

    for cls, xc, yc, bw, bh in boxes:
        # Convert normalized YOLO coords → pixel coords
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # --- Filled transparent rectangle ---
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # --- Label background ---
        label = str(cls)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # --- Outline ---
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)

        # --- Label text ---
        cv2.putText(debug_img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Blend overlay → debug_img
    cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)

    return debug_img



# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------
def process_dataset(root_dir, output_dir, debug=True, verbose=True, factor=90 ):
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

        # Load image
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]

        # Load YOLO labels
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                boxes.append([cls, xc, yc, bw, bh])

        # Rotate image + boxes
        angle_cw = factor
        rotated_img, M, new_w, new_h = rotate_img(img, angle_cw)
        rotated_boxes = rotate_yolo_boxes_no_crop(boxes, M, orig_w, orig_h, new_w, new_h)
        
        addtofname = '_rotate' + f'{factor}'
        #now create a new fname 
        new_fname = os.path.splitext(fname)[0] + addtofname + os.path.splitext(fname)[1]
        out_img_path = os.path.join(out_img_dir, new_fname )
        cv2.imwrite(out_img_path, rotated_img)

        # Save processed labels
        out_txt_path = os.path.join(out_lbl_dir, os.path.splitext(new_fname)[0] + ".txt")
        with open(out_txt_path, "w") as f:
            for cls, xc, yc, bw, bh in rotated_boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if debug: 
            # Debug image with drawn boxes
            debug_img = draw_boxes(rotated_img, rotated_boxes)
            cv2.imwrite(os.path.join(debug_img_dir, new_fname), debug_img)

            # Copy labels to debug folder
            with open(os.path.join(debug_lbl_dir, os.path.splitext(new_fname)[0] + ".txt"), "w") as f:
                for cls, xc, yc, bw, bh in rotated_boxes:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if verbose:
            print(f"Processed {fname}")


# ---------------------------------------------------------
# Main entry from __init__ 
# ---------------------------------------------------------
def rotateImg_main (root_dir, output_dir, debug=False, verbose=True, factor=90):
    if verbose: 
        print(f'>> rotate for : {root_dir}')
    
    # process dataset for rotate is different from the others 
    process_dataset(root_dir = root_dir,
                    output_dir = output_dir,
                    debug = debug,
                    verbose = verbose,
                    factor = factor)
    
    if verbose: 
        print(f'>> rotate completed ')

