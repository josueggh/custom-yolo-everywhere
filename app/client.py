import os
import subprocess
import inquirer

def list_config_files():
    """List all JSON files in the current directory that can serve as configuration files.
    Optionally, you might want to filter out some default files.
    """
    # Here we list all .json files. Adjust filtering as needed.
    return [f for f in os.listdir("./configs") if f.endswith(".json")]

def create_new_config():
    """Call the client script to create a new configuration."""
    try:
        subprocess.run(["python", "app/config.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating new configuration: {e}")

def prepare_data(config_file):
    """Call the client script to create a new configuration."""
    try:
        subprocess.run(["python", "app/prepare.py",  "--config", "./configs/"+config_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating new configuration: {e}")

def train_model(config_file):
    """Run train.py with the selected configuration file."""
    try:
        subprocess.run(["python", "app/train.py", "--config", "./configs/"+config_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")

def export_model(config_file):
    """Run export.py with the selected configuration file."""
    try:
        subprocess.run(["python", "app/export.py", "--config", "./configs/"+config_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during model export: {e}")

def main_menu():
    while True:
        config_files = list_config_files()
        # If no config files, only show the "Create" option.
        if not config_files:
            choices = ["Create a new configuration", "Quit"]
        else:
            choices = [
                "Create a new configuration",
                "Prepare dataset",
                "Train based on configuration",
                "Export model to tfjs",
                "Quit"
            ]
        answer = inquirer.prompt([
            inquirer.List('option', message="Select an option", choices=choices)
        ])
        if not answer:
            break
        option = answer["option"]

        if option == "Create a new configuration":
            create_new_config()
        elif option == "Prepare dataset":
            config_files = list_config_files()
            if not config_files:
                print("No configuration files found. Please create one first.")
                continue
            selected = inquirer.prompt([
                inquirer.List('config_file', message="Select a configuration file for training", choices=config_files)
            ])
            if selected and selected.get("config_file"):
                prepare_data(selected["config_file"])
        elif option == "Train based on configuration":
            config_files = list_config_files()
            if not config_files:
                print("No configuration files found. Please create one first.")
                continue
            selected = inquirer.prompt([
                inquirer.List('config_file', message="Select a configuration file for training", choices=config_files)
            ])
            if selected and selected.get("config_file"):
                train_model(selected["config_file"])
        elif option == "Export model to tfjs":
            config_files = list_config_files()
            if not config_files:
                print("No configuration files found. Please create one first.")
                continue
            selected = inquirer.prompt([
                inquirer.List('config_file', message="Select a configuration file for export", choices=config_files)
            ])
            if selected and selected.get("config_file"):
                export_model(selected["config_file"])
        elif option == "Quit":
            print("Exiting...")
            break

if __name__ == "__main__":
    main_menu()