import os
import glob
from pathlib import Path

def polygon_to_bbox_normalized(points):
    xs = points[0::2]
    ys = points[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return cx, cy, w, h

def convert_label_file(label_path):
    new_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            coords = list(map(float, parts[1:]))

            cx, cy, w, h = polygon_to_bbox_normalized(coords)

            new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return new_lines

def PolyToBB(label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for label_path in glob.glob(f"{label_dir}/*.txt"):
        new_labels = convert_label_file(label_path)

        out_path = f"{out_dir}/{Path(label_path).name}"
        with open(out_path, "w") as f:
            f.writelines(new_labels)

        print(f"Converted {label_path} → {out_path}")

# Example:
# PolyToBB("labels_poly", "labels_yolo")