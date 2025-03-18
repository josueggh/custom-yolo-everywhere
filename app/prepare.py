import os
import json
import shutil
import argparse
from  utils.labelstudio_export import export_annotations
from utils.dataset_preparation import (
    merge_extracted_exports,
    split_dataset_from_extracted,
    generate_data_yaml
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def main(config_path):
    config = load_config(config_path)
    api_key = config["api_key"]
    projects = config["projects"]
    valid_tags = config["tags"]
    output_dataset_dir = config["output_dataset_dir"]
    train_ratio = config.get("train_ratio", 0.8)
    export_format = config.get("export_format", "YOLO_OBB_WITH_IMAGES")

    extracted_paths = []

    for project in projects:
        project_id = project["id"]
        print(f"Exporting annotations for project {project_id}...")
        try:
            result = export_annotations(project_id, api_key, export_format)
            if isinstance(result, str):
                extracted_paths.append(result)
            else:
                print("Unsupported JSON export - please use ZIP export for now.")
        except Exception as e:
            print(e)

    if not extracted_paths:
        print("No export data found. Please check your Label Studio projects and export format.")
        return

    # If there are multiple exports, merge them; otherwise, use the single extracted directory.
    if len(extracted_paths) > 1:
        merged_dir = "merged_exports"
        print(f"Merging exports into {merged_dir}...")
        merge_extracted_exports(extracted_paths, merged_dir)
        extracted_dir = merged_dir
    else:
        extracted_dir = extracted_paths[0]

    # Split the dataset into train and validation sets
    split_dataset_from_extracted(extracted_dir, output_dataset_dir, train_ratio)

    # Generate the data.yaml file
    data_yaml_path = os.path.join(output_dataset_dir, "data.yaml")
    generate_data_yaml(output_dataset_dir, data_yaml_path, nc=len(valid_tags), names=valid_tags)

    # Cleanup: remove the merged directory if it was created
    if len(extracted_paths) > 1 and os.path.exists(merged_dir):
        print(f"Removing merged directory: {merged_dir}")
        shutil.rmtree(merged_dir)

    # Also remove each individual export directory if they still exist
    for path in extracted_paths:
        if os.path.exists(path):
            print(f"Removing temporary export directory: {path}")
            shutil.rmtree(path)

    print("Dataset automation complete. You can now run yolo-custom/train.py to train your model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data using a configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "config.json"),
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    main(args.config)