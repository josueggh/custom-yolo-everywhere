import os
import json
import inquirer
import requests

API_KEY_FILE = ".apikey"
CONFIGS_DIR = "configs"

def get_saved_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()
    return None


def save_api_key(key):
    with open(API_KEY_FILE, "w") as f:
        f.write(key)


def get_label_studio_projects(api_key, base_url):
    headers = {"Authorization": f"Token {api_key}"}
    url = f"{base_url}/api/projects"
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        else:
            return data
    else:
        raise Exception(f"Failed to fetch projects: {response.status_code} {response.text}")


def get_project_detail(api_key, base_url, project_id):
    headers = {"Authorization": f"Token {api_key}"}
    url = f"{base_url}/api/projects/{project_id}"
    response = requests.get(url, headers=headers)
    if response.ok:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch project detail for project {project_id}: {response.status_code} {response.text}")


def main():
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")

    saved_api_key = get_saved_api_key()
    if saved_api_key:
        print(f"Using saved API key: {saved_api_key}")
        change = inquirer.prompt(
            [inquirer.Confirm('change_key', message="Would you like to change the API key?", default=False)])
        if change and change.get('change_key'):
            api_key = input("Enter your new Label Studio API Key: ")
            save_api_key(api_key)
        else:
            api_key = saved_api_key
    else:
        api_key = input("Enter your Label Studio API Key: ")
        save_api_key(api_key)

    try:
        projects = get_label_studio_projects(api_key, base_url)
    except Exception as e:
        print("Error fetching projects:", e)
        return

    if not projects:
        print("No projects found in Label Studio!")
        return

    choices = [
        (f"{proj.get('title', 'Unnamed Project')} (ID: {proj.get('id')})", proj.get('id'))
        for proj in projects
        if isinstance(proj, dict) and proj.get('id') is not None
    ]
    if not choices:
        print("No valid project data found!")
        return

    project_answer = inquirer.prompt([
        inquirer.List('project_id',
                      message="Select a project",
                      choices=choices)
    ])
    if not project_answer:
        print("No answer provided. Exiting.")
        return

    selected_project_id = project_answer["project_id"]
    try:
        selected_project_detail = get_project_detail(api_key, base_url, selected_project_id)
    except Exception as e:
        print("Error fetching project detail:", e)
        return

    selected_project_name = selected_project_detail.get("title", "default_project")
    # Retrieve tags from the parsed label configuration.
    project_tags = selected_project_detail.get("parsed_label_config", {}).get("label", {}).get("labels", [])

    if not project_tags:
        print("No tags found in the selected project. Please enter tags manually.")
        tag_question = inquirer.Text('tags',
                                     message="Enter tags to use (comma separated)",
                                     default="penguin,turtle")
    else:
        tag_question = inquirer.Checkbox('selected_tags',
                                         message="Select tags to use (use space to select, enter to finish)",
                                         choices=project_tags,
                                         default=project_tags)

    additional_questions = [
        tag_question,
        inquirer.Text('train_ratio',
                      message="Enter train ratio (0 to 1)",
                      default="0.8"),
        inquirer.Text('epochs',
                      message="Enter number of training epochs",
                      default="100"),
        inquirer.Text('config_filename',
                      message="Enter filename for the new configuration",
                      default="new_config")
    ]
    additional_answers = inquirer.prompt(additional_questions)
    if not additional_answers:
        print("No additional answers provided. Exiting.")
        return

    if "selected_tags" in additional_answers:
        selected_tags = additional_answers["selected_tags"]
    else:
        selected_tags = [tag.strip() for tag in additional_answers["tags"].split(",")]

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_filename = additional_answers["config_filename"]
    config_path = os.path.join(CONFIGS_DIR, config_filename + ".json")

    new_config = {
        "api_key": api_key,
        "projects": [
            {"id": selected_project_id, "name": selected_project_name}
        ],
        "tags": selected_tags,
        "export_format": "YOLO_OBB_WITH_IMAGES",
        "config_filename": config_filename,
        "output_dataset_dir": "yolo_custom/"+config_filename,
        "train_ratio": float(additional_answers["train_ratio"]),
        "training": {
            "project": "yolo_custom/"+config_filename+"/model",
            "experiment_name": f"experiment_{selected_project_id}",
            "epochs": int(additional_answers["epochs"]),
            "imgsz": 640,
            "weights": "yolov8n.pt"
        }
    }

    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=4)
    print(f"New configuration saved to {config_path}")


if __name__ == "__main__":
    main()