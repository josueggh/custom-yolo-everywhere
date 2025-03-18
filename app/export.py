import json
import os
import argparse
from ultralytics import YOLO

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def main(config_path):
    config = load_config(config_path)
    training_config = config.get("training", {})
    output_dataset_dir = config["output_dataset_dir"]

    project = training_config.get("project", "default_project")
    experiment_name = training_config.get("experiment_name", "default_experiment")

    # Assume the trained model is saved in: ./<project>/<experiment_name>/weights/best.pt
    model_path = os.path.join(project, experiment_name, "weights", "best.pt")
    print(f"Loading model from: {model_path}")

    model = YOLO(model_path)
    # Export the model to TFJS format
    model.export(format='tfjs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export YOLO model to TensorFlow.js format using a configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("..", "config.json"),
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    main(args.config)