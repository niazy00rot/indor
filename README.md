#Indoor Scene Classification
##Overview
Deep learning-based indoor scene classification using PyTorch and ResNet50 on the Indoor CVPR 2019 dataset (67 categories).

##Quick Start
Upload kaggle.json when prompted

Run all cells in order

Model trains automatically with GPU acceleration

##Features
67 indoor scene categories (bedroom, kitchen, subway, etc.)

ResNet50 with transfer learning

Data augmentation & visualization

Training/validation metrics

Confusion matrix & evaluation

##Requirements
Python 3.7+

PyTorch, torchvision

Kaggle API (for dataset download)

GPU recommended

##Files
Indoor_Scene_Classification.ipynb - Main notebook

kaggle.json - Kaggle credentials (upload manually)

Dataset auto-downloads to indoor_scenes/

##Training
Epochs: 25

Batch size: 32

Optimizer: Adam (lr=0.001)

Loss: CrossEntropyLoss

##Results
Real-time accuracy tracking

Per-class performance metrics

Visualization of predictions

Note: Requires Kaggle account and GPU for optimal performance.
