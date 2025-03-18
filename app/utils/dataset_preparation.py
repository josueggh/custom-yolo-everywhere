import os
import random
import shutil


def merge_extracted_exports(extracted_paths, merged_dir):
    """
    Merge multiple extracted directories (each with subdirectories 'images' and 'labels')
    into a single directory.
    """
    images_merged = os.path.join(merged_dir, "images")
    labels_merged = os.path.join(merged_dir, "labels")
    os.makedirs(images_merged, exist_ok=True)
    os.makedirs(labels_merged, exist_ok=True)

    for path in extracted_paths:
        images_dir = os.path.join(path, "images")
        labels_dir = os.path.join(path, "labels")
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                src = os.path.join(images_dir, file)
                dst = os.path.join(images_merged, file)
                shutil.copy2(src, dst)
        if os.path.exists(labels_dir):
            for file in os.listdir(labels_dir):
                src = os.path.join(labels_dir, file)
                dst = os.path.join(labels_merged, file)
                shutil.copy2(src, dst)
    print(f"Exports merged into {merged_dir}")


def split_dataset_from_extracted(extracted_dir, output_dataset_dir, train_ratio=0.8):
    """
    Given an extracted directory (with subdirectories 'images' and 'labels'),
    randomly split the images (and corresponding label files) into training and validation sets.
    """
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = create_dirs(output_dataset_dir)
    images_dir = os.path.join(extracted_dir, "images")
    labels_dir = os.path.join(extracted_dir, "labels")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_list = image_files[:split_index]
    val_list = image_files[split_index:]

    for image_filename in train_list:
        src_img = os.path.join(images_dir, image_filename)
        dst_img = os.path.join(train_images_dir, image_filename)
        shutil.copy2(src_img, dst_img)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_filename)
        if os.path.exists(src_label):
            dst_label = os.path.join(train_labels_dir, label_filename)
            shutil.copy2(src_label, dst_label)

    for image_filename in val_list:
        src_img = os.path.join(images_dir, image_filename)
        dst_img = os.path.join(val_images_dir, image_filename)
        shutil.copy2(src_img, dst_img)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_filename)
        if os.path.exists(src_label):
            dst_label = os.path.join(val_labels_dir, label_filename)
            shutil.copy2(src_label, dst_label)

    print(f"Dataset created: {len(train_list)} training items, {len(val_list)} validation items.")


def generate_data_yaml(output_dataset_dir, data_yaml_path, nc, names):
    """
    Generates a data.yaml file with absolute paths for the train and val image directories,
    the number of classes (nc), and the class names (names) in valid YAML format.
    """
    train_images_path = os.path.join(os.path.abspath(output_dataset_dir), "train", "images")
    val_images_path = os.path.join(os.path.abspath(output_dataset_dir), "val", "images")
    # Create a YAML list for names
    names_yaml = "\n".join(["  - " + name for name in names])
    yaml_content = f'''train: "{train_images_path}"
val: "{val_images_path}"

nc: {nc}
names:
{names_yaml}
'''
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml generated at {data_yaml_path}")
def create_dirs(base_dir):
    """
    Create the necessary directory structure for train and val (images and labels).
    """
    train_images = os.path.join(base_dir, "train", "images")
    train_labels = os.path.join(base_dir, "train", "labels")
    val_images = os.path.join(base_dir, "val", "images")
    val_labels = os.path.join(base_dir, "val", "labels")
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    return train_images, train_labels, val_images, val_labels

