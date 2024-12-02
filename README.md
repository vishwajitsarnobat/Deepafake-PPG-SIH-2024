# Deepfake-SIH-2024
Official repository for SIH 2024, including codes for deepfake detection

# Deepfake Source Detection

This project uses machine learning to detect whether a video is real or manipulated using deepfake techniques. It extracts biological signals from facial features and uses a **ResNet-50** model to classify videos into real or fake categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Details](#model-details)
- [Results](#results)

---

## Project Overview

The goal of this project is to classify videos as **real** or **fake** based on facial signals. Real videos are taken from the `original_sequences` folder, and manipulated videos (deepfakes) are taken from the `manipulated_sequences` folder. The system extracts **PPG signals** from faces and uses them to train a **ResNet-50** model.

## Dataset

The dataset is organized into:

- **original_sequences**: Real videos (labeled `0`).
- **manipulated_sequences**: Fake videos (labeled `1`).

Each video is processed, faces are extracted, and **PPG signals** are used for training the classification model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/deepfake-source-detection.git
    cd deepfake-source-detection
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv deepfake_venv
    source deepfake_venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the entire pipeline with the following command:
```bash
python3 main.py
```

This will:

    Create the labels (create_labels.py).
    Preprocess videos and extract faces (preprocess_videos.py).
    Extract PPG signals and create cells (signal_extraction.py).
    Train the model (train.py).
    Evaluate the model (evaluate.py).

File Structure

deepfake_source_detection/
│
├── dataset/                                # Your dataset
│   ├── manipulated_sequences/             # Fake videos
│   ├── original_sequences/                # Real videos
│
├── src/                                   # Source code
│   ├── create_labels.py                  # Create labels
│   ├── preprocess_videos.py              # Extract faces and frames
│   ├── signal_extraction.py              # Generate PPG cells
│   ├── model.py                          # ResNet-50 model definition
│   ├── train.py                          # Train the model
│   ├── evaluate.py                       # Evaluate the model
│
├── configs/                              # Configurations
│   ├── config.yaml                       # Config file for paths and parameters
│
├── data/                                 # Processed data (generated)
│   ├── labels.csv                        # Generated labels
│   ├── frames/                           # Extracted frames
│   ├── ppg_cells/                        # Generated PPG cells
│
├── results/                              # Results folder
│   ├── models/                           # Saved models
│
├── requirements.txt                      # Dependencies
├── README.md                             # Project documentation
└── main.py                               # Pipeline entry point

Model Details

The model used for classification is ResNet-50 (pre-trained on ImageNet). It is fine-tuned on PPG cells extracted from video frames for binary classification (real vs fake).
Training Parameters:

    Batch Size: 16
    Learning Rate: 0.001
    Epochs: 10 (adjustable in config.yaml)

Evaluation:

The model is evaluated based on accuracy.
Results

After training, the model’s performance is evaluated using accuracy (percentage of correct classifications).
License

This project is licensed under the MIT License.


---

This version is **properly formatted** for Markdown, and when you paste it into your `README.md` file, it should display as expected with headings, bullet points, and code blocks.

Let me know if you need further help!


