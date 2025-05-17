# Image Restoration Through Real Noise Modeling

Implementation of a VAE-based image restoration technique for degraded photos that learns from unpaired clean and noisy images. Improves on the original method (Wan et al. 2022) by replacing transposed convolution layers with resize-convolution blocks to eliminate checkerboard artifacts.

## About

Based on the paper:
> WAN, Ziyu, ZHANG, Bo, CHEN, Dongdong, et al. "Old photo restoration via deep latent space translation." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022. [arXiv:2009.07047](https://arxiv.org/abs/2009.07047)

## Project Structure

- `src/train.py` - Script for training different modules/models
- `src/inference.py` - Script for denoising images using a trained model
- `src/evaluate.py` - Functions for model evaluation (can be modified for custom evaluation)
- `src/models.py` - Implementation of model classes (VAE1, VAE2, and Mapping)
- `src/networks.py` - Shared network structures
- `src/dataset.py` - Dataset builders and dataloaders
- `src/utils.py` - Image processing and evaluation utilities
- `src/config.py` - Configuration parameters for training and evaluation

## Installation

The code has been tested with Python 3.8.0. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Training

The model consists of three modules that must be trained separately:
1. VAE1 and VAE2 can be trained independently
2. The mapping module requires pre-trained VAE1 and VAE2 models

Training each module:

```bash
# Train VAE1
python src/train.py --cfg-path configs/vae.yaml --stage vae1 --output-path checkpoints/vae1.ckpt

# Train VAE2
python src/train.py --cfg-path configs/vae.yaml --stage vae2 --output-path checkpoints/vae2.ckpt

# Train mapping (requires trained VAE1 and VAE2)
python src/train.py --cfg-path configs/mapping.yaml --stage mapping --output-path checkpoints/full.ckpt
```

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

### Inference

Run inference on images using a trained model:

```bash
python src/inference.py -i <input_folder> -o <output_folder> -m <fullmodel_path>
```

**Note:** The model is not optimized for processing large images and may result in out-of-memory errors with high-resolution inputs.
