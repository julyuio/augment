import cv2
import os

def load_yolo_annotations(txt_path):
    boxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    cls, xc, yc, w, h = box

    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)

    return cls, x1, y1, x2, y2


def draw_boxes(image_path, txt_path, output_path="output.jpg"):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_h, img_w = img.shape[:2]

    # Load YOLO annotations
    boxes = load_yolo_annotations(txt_path)

    # Draw each bounding box
    for box in boxes:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class label
        cv2.putText(
            img,
            f"class {cls}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save output
    cv2.imwrite(output_path, img)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    
    # Example usage
    draw_boxes("train/images/703.jpg", "train/labels/703.txt", "result.jpg")