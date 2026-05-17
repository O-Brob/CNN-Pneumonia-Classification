# Convolutional Neural Network - Pneumonia X-ray Classification

TODO

## Overview
TODO

## Table of Contents
TODO

## Features
TODO

## Project Structure
````
CNN-Pneumonia-Classification/
├── data/                   (Dataset download target folder)
│   └── curl_data.sh        (Shell file to facilitate download)
├── models/                 (Trained model target folder)
│   └── .gitignore          (Ensure folder exists & prevent upload)
├── src/                    (Folder of all Python source code)
│   ├── __init__.py         (Marks directory as Python package)
│   ├── config.py           (Centralized hyperparameters & paths)
│   ├── data_loader.py      (Transforms & Preprocessing of Dataset)
│   ├── infer.py            (Perform inference on singular x-ray)
│   ├── model.py            (CNN architecture definition)
│   └── trainer.py          (Trains, validates, evaluates model)
├── .gitignore              (Main .gitignore of non-source files)
├── LICENSE.md              (Legal usage/distribution terms)
├── main.py                 (CLI entry point for the program)
├── README.md               (Project description & instructions)
├── requirements.txt        (Python dependencies, e.g. Torch)
├── RESULTS.md              (Performance metrics & evaluation logs)
└── third-party-licenses.md (Attributions for external libs & data)
````

## Requirements
For this project to function as intended, there are certain software and hardware requirements that need to be met.

- 1.) Software & Dependencies
    - **Python**: (Version 3.10+)

    - **External Packages**: All necessary external packages are listed in the `requirements.txt` file.

    To install all required dependencies, run:
    ````
    pip install -r requirements.txt
    ````
- 2.) Hardware Recommendations
    - **GPU**: A CUDA-capable NVIDIA GPU is recommended for training the CNN model. Training on a CPU is possible, and will be done automatically if no CUDA-capable GPU is found, but is *significantly* slower.

    - **VRAM**: At least 4GB of VRAM is needed to comfortably handle standard batch sizes during training in this project. With the default configuration of `src/config.py`, a use of approximately 3.8GB of VRAM was measured on my system.

- 3.) Dataset & Permission Compliance
    - **Data Access**: Execution of the `data/curl_data.sh` script for downloading the dataset prior to unpacking it requires an active internet connection, and more notably, `curl` installed on your system.

    - **Licensing**: Ensure you comply with the terms specified in `LICENSE.md` regarding the project's source files, and `third-party-licenses.md` regarding the use and distribution of the lung X-ray dataset that is used.

## Installation
TODO

## User Guide
TODO

### Downloading Dataset
TODO

### Training 
TODO

### Evaluation
TODO

### Inference
TODO

## Model & Dataset
TODO

## License
TODO