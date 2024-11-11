# Seaturtle
This project focuses on segmenting three parts of sea turtles in images(Head, Flippers, Carapace) using deep learning. It leverages PyTorch and uses SeaTurtleID2022 datasets. The models are trained and evaluated using Intersection over Union (IoU) metrics.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Evaluation](#evaluation)
- [Results and Visualization](#results-and-visualization)
- [Usage](#usage)

## Project Overview
This project implements a baseline model and two deep learning models to perform semantic segmentation on sea turtle images. Key components include:

Custom Dataset Preparation using COCO annotations.
Data Augmentation for improved generalization.
Three Segmentation Models:

- Traditional method
- UNet-based Model
- DeepLabV3

## Dataset
The dataset is organized with COCO-style annotations, and is split into training, validation, and test sets. Images, metadata and annotations files are stored in Google Drive.

`Image Directory: /content/drive/MyDrive/9517`  
`Annotations File: /content/drive/MyDrive/9517/annotations.json`  
`Metadata File: /content/drive/MyDrive/9517/metadata_splits.csv`

## Model Architecture
1. DeepLabV3 with Attention
  Backbone: ResNet101 pretrained on ImageNet.
  ASPP Module: Atrous Spatial Pyramid Pooling with multiple dilation rates.
  Self-Attention Module: Adds spatial attention to enhance the feature representation for segmentation.
  Classifier: Final 1x1 convolution for class predictions.
2. UNet-based Model
  Encoder: ResNet34-based encoder layers.
  Decoder: Upsampling layers with skip connections for feature reuse.
  Output Layer: 1x1 convolution to produce segmentation masks.
3. Traditional Method
  This method uses traditional image processing techniques as a baseline for comparison with deep learning methods.


## Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- PIL
- tqdm
- matplotlib
- pycocotools

To install the required packages, run:

```bash
pip install torch torchvision pandas pillow tqdm matplotlib pycocotools
```

## Evaluation

For all models, including the traditional method, performance is evaluated using **Intersection over Union (IoU)** for each class:

- **Head**
- **Flippers**
- **Carapace**

## Results and Visualization
IoU Scores: Provides IoU scores for each class on the test set.
Prediction Visualization: The predict_and_visualize function generates sample predictions, showing the original image, ground truth mask, and predicted mask.

To evaluate and visualize the model's performance, you can use the following code:

```python
evaluate_iou_on_test_set(model, test_loader, device)
print("Visualizing predictions on test data...")
fig_test = predict_and_visualize(model, test_loader, device)
plt.show()
```

## Usage
1. Mount Google Drive: The dataset is stored on Google Drive and mounts it for access.
2. Prepare Dataset: Initialize the SeaTurtleDataset with image paths and annotations.
3. Train the Model: Use the Trainer class to train the deep learning models.
4. Evaluate IoU: show IoU evaluation on the test set.
