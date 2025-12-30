import os
import cv2
import numpy as np

# ---------------------------------------------------------
# ROTATE IMAGE WITHOUT CROPPING
# ---------------------------------------------------------
def rotate_image_no_crop(img, angle_cw):
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
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    debug_img = img.copy()

    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(debug_img, str(cls), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return debug_img


# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------
def process_dataset(root_dir, output_dir, angle_cw):
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
        rotated_img, M, new_w, new_h = rotate_image_no_crop(img, angle_cw)
        rotated_boxes = rotate_yolo_boxes_no_crop(boxes, M, orig_w, orig_h, new_w, new_h)

        # Save rotated image
        out_img_path = os.path.join(out_img_dir, fname)
        cv2.imwrite(out_img_path, rotated_img)

        # Save rotated labels
        out_txt_path = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")
        with open(out_txt_path, "w") as f:
            for cls, xc, yc, bw, bh in rotated_boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Debug image with drawn boxes
        debug_img = draw_boxes(rotated_img, rotated_boxes)
        cv2.imwrite(os.path.join(debug_img_dir, fname), debug_img)

        # Copy labels to debug folder
        with open(os.path.join(debug_lbl_dir, os.path.splitext(fname)[0] + ".txt"), "w") as f:
            for cls, xc, yc, bw, bh in rotated_boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"Processed {fname}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    process_dataset(
        root_dir="train",
        output_dir="augmented_30deg",
        angle_cw=30  # change angle here
    )