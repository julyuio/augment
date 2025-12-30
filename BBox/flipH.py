import os
import cv2
import numpy as np

# ---------------------------------------------------------
# HORIZONTAL FLIP IMAGE
# ---------------------------------------------------------
def flip_image_horizontal(img):
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
def process_dataset(root_dir, output_dir):
    addtofname =  '_flipH' #adding to filename 
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

        # Load YOLO labels
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                boxes.append([cls, xc, yc, bw, bh])

        # Flip image + boxes
        flipped_img = flip_image_horizontal(img)
        flipped_boxes = flip_yolo_boxes_horizontal(boxes)

        # Save flipped image
        new_fname = os.path.splitext(fname)[0] + addtofname + os.path.splitext(fname)[1]
        out_img_path = os.path.join(out_img_dir, new_fname )
        cv2.imwrite(out_img_path, flipped_img)

        # Save flipped labels
        out_txt_path = os.path.join(out_lbl_dir, os.path.splitext(new_fname)[0]  + ".txt")
        with open(out_txt_path, "w") as f:
            for cls, xc, yc, bw, bh in flipped_boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Debug image with drawn boxes
        debug_img = draw_boxes(flipped_img, flipped_boxes)
        cv2.imwrite(os.path.join(debug_img_dir, new_fname) , debug_img)

        # Copy labels to debug folder
        with open(os.path.join(debug_lbl_dir, os.path.splitext(new_fname)[0] + ".txt"), "w") as f:
            for cls, xc, yc, bw, bh in flipped_boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"Processed {fname}")


def flipH_dataset (root_dir,output_dir):
    print(f'>> Fliping along horizontal : {root_dir}')
    process_dataset(root_dir, output_dir)
    print(f'>> flipH completed ')


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    process_dataset(
        root_dir="train",
        output_dir="train_flipped_H"
    )
