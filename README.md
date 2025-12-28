# Indoor Scene Classification

## Overview
Deep learning model for classifying indoor scenes using PyTorch and ResNet50.  
Trained on the **Indoor CVPR 2019** dataset (67 scene categories).

## Quick Start
1. **Upload** `kaggle.json` when prompted
2. **Run all cells** in sequence
3. **Model trains automatically** with GPU support

## Features
- **67 indoor scene categories** (bedroom, kitchen, subway, bookstore, etc.)
- **ResNet50** with transfer learning
- **Data augmentation** and visualization tools
- **Training metrics** tracking (loss/accuracy)
- **Evaluation**: confusion matrix & classification report

## Requirements
- Python 3.7+
- PyTorch & torchvision
- Kaggle API key (`kaggle.json`)
- GPU recommended (T4/P100)

# Project Structure

## Training Configuration
- **Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam (LR=0.001)
- **Loss Function**: CrossEntropy

## Output
- Real-time training progress
- Accuracy/Loss plots
- Per-class performance metrics
- Sample predictions visualization

---

> **Note**: Requires Kaggle account and GPU for optimal performance.
