import os
import cv2
import numpy as np

# ---------------------------------------------------------
# ADD RANDOM GAUSSIAN NOISE
# ---------------------------------------------------------
def add_random_noise(img, mean=0, std=25):
    """
    Adds Gaussian noise to an image.
    mean: noise mean
    std: noise standard deviation (higher = stronger noise)
    """
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


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
def process_dataset(root_dir, output_dir, noise_std=25):
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

        # Add noise
        noisy_img = add_random_noise(img, std=noise_std)

        # Save noisy image
        out_img_path = os.path.join(out_img_dir, fname)
        cv2.imwrite(out_img_path, noisy_img)

        # Save labels (unchanged)
        out_txt_path = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")
        with open(out_txt_path, "w") as f:
            for cls, xc, yc, bw, bh in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Debug image with drawn boxes
        debug_img = draw_boxes(noisy_img, boxes)
        cv2.imwrite(os.path.join(debug_img_dir, fname), debug_img)

        # Copy labels to debug folder
        with open(os.path.join(debug_lbl_dir, os.path.splitext(fname)[0] + ".txt"), "w") as f:
            for cls, xc, yc, bw, bh in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"Processed {fname}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    process_dataset(
        root_dir="train",
        output_dir="train_noised",
        noise_std=25  # adjust noise strength here
    )