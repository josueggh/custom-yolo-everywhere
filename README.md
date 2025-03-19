# Custom YOLO Everywhere

<img src="https://firebasestorage.googleapis.com/v0/b/josue-ggh.firebasestorage.app/o/covers%2Fheader.png?alt=media&token=f6f0b4cf-bdb3-4abf-a581-223980a92f3e  "/>

![License](https://img.shields.io/github/license/josueggh/custom-yolo-everywhere?logo=github)
![Issues](https://img.shields.io/github/issues/josueggh/custom-yolo-everywhere)

Custom YOLO Everywhere is an open-source solution that bridges the gap between image annotation and custom object
detection model deployment. This project seamlessly integrates Label Studio with a YOLO client, automating the entire
workflow from annotation export to model training and multi-format export. Whether you want to deploy on the web using
TensorFlow.js, on mobile, or other native platforms via TFLite, this tool empowers developers, researchers, and
hobbyists to build and deploy custom YOLO models.

### Video Tutorial 🎥  
For a step-by-step guide on setting up and using **Custom YOLO Everywhere**, check out our tutorial:  
📺 [Watch the video on YouTube](https://www.youtube.com/watch?v=sAkxTy7Vy6w)  


## Prerequisites

- **Docker & Docker Compose:** to run Label Studio.
- **Python (3.10+):** for training and exporting the YOLO model

## Docker Setup

This project uses Docker Compose to manage the services:

Label Studio: Runs on port 8080.

### .env File

Create a `.env` file in the root with your environment variables:

```bash
LABEL_STUDIO_URL=http://localhost:8080
```

`Note`: The env.template can be used as reference

### Running the Containers

1. Build and start the containers:

```bash
docker-compose up --build -d
```

Label Studio will be available at http://localhost:8080. Use it to annotate your images. When you export your
annotations to YOLO format from Label Studio, save the files into the yolo-custom/dataset/train and
yolo-custom/dataset/val directories (or run an automation script, see below).

3. Running the Interactive Client

Install the dependencies

```bash
pip install -r requirements.txt
```

Now you can run the interactive client from the project root by executing:

```bash
python app/client.py
```

# Command-Line overview

This project includes an interactive command-line client that simplifies managing configurations and running key tasks.
Once you launch the client, you will see a menu with several options:

### Create a New Configuration:

Launch an interactive process to create a new configuration file. This process will prompt you for your Label Studio API
key (you can find it under the "account & settings" sections of your Label Studio installation), then you'll be able to
select a project and its available tags, and ask for additional training parameters (e.g., train ratio and epochs). The
generated configuration is saved in the `configs/` directory.

### Prepare Dataset:

Lists all configuration files found in the `configs/` directory. After you select one, the client calls the dataset
preparation script (`app/prepare.py`) using that configuration. This process exports annotations from Label Studio,
merges multiple exports if necessary, splits the dataset into training and validation sets, and generates a proper
`data.yaml` file.

### Train Based on Configuration:

Lists all configuration files from the `configs/` directory and allows you to select one to use for training. Once
selected, it triggers the training script (`app/train.py`) with the chosen configuration as a parameter.

### Export Model to TensorfFlow Formats

Similarly, this option lists the configuration files. When you select one, it runs the export script (`app/export.py`)
with the chosen configuration file as a parameter to convert your trained model to TensorFlow format.

# Running on Apple Silicon

To run on Apple Silicon (M1/M2), create and activate an ARM-native virtual environment and install the required packages:

```bash
conda create -n arm-env python=3.10
conda activate arm-env

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python app/client.py
```

## M1 troubleshooting

Sometimes, the JAX library presents issues on Apple devices. Follow the [official tutorial](https://developer.apple.com/metal/jax/) to ensure a proper JAX installation.


Then, install an ARM-compatible version of JAX (CPU-only) from Google’s storage:

```bash
python -m pip install jax-metal
```

Test it:

```bash
python -c 'import jax; print(jax.numpy.arange(10))'
```
