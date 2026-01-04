import os
import shutil
import random

def makeTrainTestVal(train_dir, augmented_dir, output_dir):
    """
    Combines original + augmented YOLO datasets and splits them into:
    70% train, 20% val, 10% test.

    Args:
        train_dir (str): Path to original dataset (with images/ and labels/).
        augmented_dir (str): Path to augmented dataset.
        output_dir (str): Output directory for split dataset.
    """

    # Output folders
    train_out = os.path.join(output_dir, 'train')
    val_out = os.path.join(output_dir, 'val')
    test_out = os.path.join(output_dir, 'test')

    def ensure_dirs():
        for d in [train_out, val_out, test_out]:
            os.makedirs(os.path.join(d, 'images'), exist_ok=True)
            os.makedirs(os.path.join(d, 'labels'), exist_ok=True)

    def collect_files(images_dir):
        """Return list of image base names (without extension)."""
        files = []
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.splitext(f)[0])
        return files

    def copy_pair(base_name, src_img_dir, src_lbl_dir, dst_dir):
        """Copy image + label pair to destination."""
        img_src = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(src_img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_src = candidate
                break

        if img_src is None:
            return  # skip if no image found

        lbl_src = os.path.join(src_lbl_dir, base_name + '.txt')

        shutil.copy(img_src, os.path.join(dst_dir, 'images'))
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(dst_dir, 'labels'))

    # Start processing
    ensure_dirs()

    # Collect original + augmented
    original = collect_files(os.path.join(train_dir, 'images'))
    augmented = collect_files(os.path.join(augmented_dir, 'images'))

    all_files = original + augmented
    random.shuffle(all_files)

    total = len(all_files)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.2)

    train_set = all_files[:train_end]
    val_set = all_files[train_end:val_end]
    test_set = all_files[val_end:]

    print(f"Total: {total}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Copy files
    for base in train_set:
        if base in original:
            copy_pair(base, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'), train_out)
        else:
            copy_pair(base, os.path.join(augmented_dir, 'images'), os.path.join(augmented_dir, 'labels'), train_out)

    for base in val_set:
        if base in original:
            copy_pair(base, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'), val_out)
        else:
            copy_pair(base, os.path.join(augmented_dir, 'images'), os.path.join(augmented_dir, 'labels'), val_out)

    for base in test_set:
        if base in original:
            copy_pair(base, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'), test_out)
        else:
            copy_pair(base, os.path.join(augmented_dir, 'images'), os.path.join(augmented_dir, 'labels'), test_out)

    print("Dataset combined and split successfully.")


# Example usage:
# makeTrainTestVal('./train', './train_augmented', './outputFinal_dataset')