import os
import shutil



def recombine_split_dataset(train_dir, val_dir, test_dir, output_dir):
    """
    Recombines YOLO-style split dataset (train/val/test) back into a single dataset.

    Args:
        train_dir (str): Path to train folder (with images/ and labels/).
        val_dir (str): Path to val folder.
        test_dir (str): Path to test folder.
        output_dir (str): Output directory for merged dataset.
    """

    out_img = os.path.join(output_dir, "images")
    out_lbl = os.path.join(output_dir, "labels")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    def copy_all(src_dir):
        img_dir = os.path.join(src_dir, "images")
        lbl_dir = os.path.join(src_dir, "labels")

        if not os.path.exists(img_dir):
            return

        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(f)[0]

                # Copy image
                shutil.copy(
                    os.path.join(img_dir, f),
                    os.path.join(out_img, f)
                )

                # Copy label if exists
                lbl_src = os.path.join(lbl_dir, base + ".txt")
                if os.path.exists(lbl_src):
                    shutil.copy(lbl_src, os.path.join(out_lbl, base + ".txt"))

    # Process all three splits
    copy_all(train_dir)
    copy_all(val_dir)
    copy_all(test_dir)

    print(f"Dataset recombined into: {output_dir}")


def recombine_split_dataset_dir(main_dir):
    train_dir = main_dir+'/train'
    val_dir = main_dir+'/val'
    test_dir =main_dir+'/test'
    output_dir =main_dir+'_ONE'
    
    recombine_split_dataset(train_dir, val_dir, test_dir, output_dir)