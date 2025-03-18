import json
import os
import argparse
from ultralytics import YOLO

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def main(config_path):
    # Load configuration from the given file path
    print(config_path)
    config = load_config(config_path)
    training_config = config.get("training", {})
    output_dataset_dir = config["output_dataset_dir"]

    # Read training parameters
    project = training_config.get("project", "default_project")
    experiment_name = training_config.get("experiment_name", "default_experiment")
    epochs = training_config.get("epochs", 100)
    imgsz = training_config.get("imgsz", 640)
    weights = training_config.get("weights", "yolov8n.pt")

    # Initialize the model with pre-trained weights
    model = YOLO(weights)

    # Train the model using the dataset specified in data.yaml
    results = model.train(
        data=output_dataset_dir+"/data.yaml",  # Path to the generated data.yaml file
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=experiment_name,
        batch=8
    )

    # The trained weights (.pt file) will be saved under project/experiment_name.
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO model using a configuration file"
    )
    print("STARTING")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("..", "config.json"),
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    main(args.config)