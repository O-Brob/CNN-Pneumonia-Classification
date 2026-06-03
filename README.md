# Convolutional Neural Network - Pneumonia X-ray Classification
[![GitHub Badge](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=fff&style=for-the-badge)](https://github.com/O-Brob/CNN-Pneumonia-Classification)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## Overview
*`CNN-Pneumonia-Classification`* is a personal learning project consisting of a PyTorch-based Convolutional Neural Network setup for detecting pneumonia from chest X-ray images. It includes full pipelines for training, evaluation, and inference on custom inputs via command line. 

The goal of the project was to develop a neural network inspired by the VGG-16 architecture, trained to classify medical images for use as a diagnostic tool for radiologists or clinicians. For this project it was decided that pneumonia classification will be the focus, although the repository can be easily forked and adapted to other forms of medical imagery. There is also opportunity to modify the model structure or configuration parameters in an attempt to further improve the results.

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [User Guide](#user-guide)
6. [Model & Dataset](#model--dataset)
7. [License](#license)  

## Features
- **Unified CLI Entrypoint**: A single entry point, `main.py`, to handle automatic downloading, training, evaluating and inference via command line flags.

- **Centralized Configuration**: Configure all hyperparameters, batch sizes, etc. from the same file, `src/config.py`.

- **Hardware Agnostic**: Automatically detects and uses an NVIDIA GPU via CUDA if available, or falls back to using the CPU if none is detected.

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

    - **Environment**: A bash shell (Linux, MacOS, or Git Bash/WSL on Windows)

    - **External Packages**: All necessary external packages are listed in the `requirements.txt` file.

    To install all required dependencies, run:
    ````bash
    pip install -r requirements.txt
    ````
- 2.) Hardware Recommendations
    - **GPU**: A CUDA-capable NVIDIA GPU is recommended for training the CNN model. Training on a CPU is possible, and will be done automatically if no CUDA-capable GPU is found, but is *significantly* slower.

    - **VRAM**: At least 4GB of VRAM is needed to comfortably handle standard batch sizes during training in this project. With the default configuration of `src/config.py`, a use of approximately 3.8GB of VRAM was measured on my system.

- 3.) Dataset & Permission Compliance
    - **Data Access**: Execution of the `data/curl_data.sh` script for downloading the dataset prior to unpacking it requires an active internet connection, and more notably, `curl` installed on your system.

    - **Licensing**: Ensure you comply with the terms specified in `LICENSE.md` regarding the project's source files, and `third-party-licenses.md` regarding the use and distribution of the lung X-ray dataset that is used.

## Installation
Provided that the requirements above have been met, installing and subsequently running the project is simple. The most straightforward method to get the project files is via `Git` cloning, using the following command:

```bash
git clone https://github.com/O-Brob/CNN-Pneumonia-Classification.git
```

The result will be the creation of a folder named `CNN-Pneumonia-Classification` in the target directory. This folder contains the project files, as shown in [Project Structure](#project-structure). To run the project, `main.py` will be used as a unified CLI entrypoint, as described in the section below.

## User Guide
The project includes full pipelines for training, evaluation, and inference on custom inputs. All these functions of the project are accessible via the unified command line interface, `main.py`, using CLI flags. As an overview, the following flags are considered, and can be called either individually or be chained:

```
-h   --help            :   Print help information
-d   --download        :   Download x-ray dataset for training
-t   --train           :   Perform CNN training on GPU if available, else CPU
-e   --evaluate        :   Evaluate performance after training over a test set
-i   --infer [image]   :   Infer a classification on the given image 
-c   --clean           :   Remove downloaded datasets and saved models
```

A more in-depth explanation of each of the flags and how they contribute to the pipeline is provided in the following subsections. For instructions on how to execute the full pipeline, see [Executing full pipeline](#executing-full-pipeline).

### Downloading Dataset
Prior to training the model, it is required to download and process the dataset by splitting the images into folders `Normal` and `Pneumonia`. This can be done automatically via the shell script `data/curl_data.sh`. This shell script can be automatically invoked by passing the `-d` or `--download` flag to `main.py`, as follows:

```bash
python main.py --download
```

### Training 
Once the dataset has been downloaded as per the instructions above, training can be initiated in a similar fashion by passing the `-t` or `--train` flag to `main.py`, as follows:

```bash
python main.py --train
```

To modify the hyperparameter configuration used during training, refer to the file `src/config.py` and the global parameters therein.

Once training has been completed, the best model over the validation dataset will be saved as `models/trained_model.pt`. This model will be used for both evaluation and inference.

### Evaluation
To evaluate the model, it is required to have trained the model, such that `models/trained_model.pt` exists. Once this prerequisite is completed, evaluation of the model can be initated by passing the `-e` or `--evaluate` flag to `main.py`, as follows:

```bash
python main.py --evaluate
```

This will output an accuracy metric over the test dataset alongside metrics such as recall, precision, false positive rate and F1-Score. Furthermore, a confusion matrix will be output.

For reference on what to expect using the default hyperparameter configuration set in `src/config.py`, see [RESULTS.md](RESULTS.md), which shows the result of training and evaluating the developed model.

### Inference
To use single image inference, it is required to have trained the model, such that `models/trained_model.pt` exists, just as for evaluation. Once this prerequisite is completed, inference can be initiated by passing the `-i` or `--infer` flag to `main.py`, followed by an absolute or relative path to an x-ray image, as follows:

```bash
python main.py --infer path/to/image.png
```

### Executing Full Pipeline
As the CLI flags for the functions mentioned above can be chained, they can be conveniently executed one after another from a single invocation as a result. The following arguments sequentially downloads and processes the dataset, trains the model, and evaluates the model, after which inference is available:

```bash
python main.py --download --train --evaluate
```

## Model & Dataset

### Model Architechture
The classifier uses a VGG-inspired Convolutional Neural Network to be used for binary classification of pneumonia in chest X-ray images. The architecture is designed for 256x256 grayscale input images (as prepared in the data loader) and consists of:

#### Convolutional Feature Extraction:
- **Conv. Block 1:** Conv2d (1->32 filters, 3x3 kernel) -> BatchNorm -> ReLU -> MaxPool (2x2) -> Dropout

- **Conv. Block 2:** Conv2d (32->64 filters, 3x3 kernel) -> BatchNorm -> ReLU -> MaxPool (2x2) -> Dropout

- **Conv. Block 3:** Conv2d (64->128 filters, 3x3 kernel) -> BatchNorm -> ReLU -> MaxPool (2x2) -> Dropout

Batch normalization is used to stabilize training and converge faster. The use of max-pooling is intended to reduce spatial dimensions to extract the "more important" details and produce a more generalized representation of the data. Dropout is used to prevent overfitting.

#### Classification Head:
- Adaptive Average Pooling with output size 6x6 is used to preserve the learned spatial features relative to near surroundings while reducing parameter count by a lot.

- Flattening is done to a 4608-dimensional feature vector.

- Fully Connected Layer (4608 -> 512 units) with ReLU activation.

- Output Layer (512 -> 1 unit) to produce raw logits for the binary classification. Raw logits are used to improve numerical stability during training.

### Dataset
The model is trained on the Kaggle [**Chest X-Ray Images (Pneumonia)**](third-party-licenses.md#chest-x-ray-images-pneumonia) dataset, a publicly available collection of chest X-ray images of both healthy patients and patients with pneumonia. 

#### Dataset Characteristics:
- **Total Images:** 5863 X-ray images of patients aged 1-5.

- **Classes:** Normal, Pneumonia.

- **Image Format:** JPEG, 8-bitdepth with resolutions ranging up to ~2000x2000 pixels.

- **Imbalance:** Dataset contains class imbalance with more pneumonia cases than normal cases in the training set.

#### Preprocessing Pipeline:
- Images are set to grayscale and resized to 256x256 pixels to meet model input expectations.

- Pixel values normalized via Z-Score normalization.

- The data imbalance is solved via undersampling. It is also ensured that different X-rays of the same patient does not leak between datasets.

## License
This project is licensed under the MIT License.  
See the full text in the [LICENSE.md](LICENSE.md) file in the repository root.

### Third‑Party Licenses & Notices

Third-party dependencies and their licenses are listed in [third-party-licenses.md](third-party-licenses.md).