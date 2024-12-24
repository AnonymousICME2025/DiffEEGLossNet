# DiffEEGLossNet: Enhancing Motor Imagery EEG Generation with Single-Step Diffusion Models

This repository contains the implementation of **DiffEEGLossNet**, a novel framework for generating high-quality EEG signals for motor imagery tasks. The work is based on diffusion probabilistic models and incorporates neuroscientific constraints to enhance the fidelity of generated EEG signals. This project accompanies our paper submitted to ICME 2025.

## Key Features
- **Motor Imagery-Specific Loss**: Integrates ERD/ERS-informed loss to ensure generated signals align with neuroscientific characteristics.
- **Progressive Distillation**: Accelerates inference through a single-step diffusion model (DiffEEGLossNet 1×).
- **High-Quality Data Generation**: Demonstrates superior fidelity and diversity metrics (IS, FID, sFID) compared to existing models.
- **Data Augmentation**: Improves classification performance on EEGNet and EEGProgress models using generated samples.

## Architecture Overview
The DiffEEGLossNet framework consists of three main modules:
1. **EEG Signal Preprocessing**: Band-pass filtering, down-sampling, and normalization of EEG signals.
2. **Diffusion Inference**: Utilizes bidirectional dilated convolutions and residual blocks for generating high-quality EEG signals.
3. **Motor Imagery Loss Integration**: Incorporates ERD/ERS patterns as part of the training loss to ensure neuroscientific fidelity.

For detailed methodology, please refer to our [paper](link-to-paper).

## Getting Started

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.9.0
- Additional libraries: NumPy, Matplotlib, Scipy

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/DiffEEGLossNet.git
cd DiffEEGLossNet

