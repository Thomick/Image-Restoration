from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
from pathlib import Path
import os
import numpy as np

import torchvision.utils as vutils

params = {
    "lr": 2e-3,
    "a_reconst": 10,
    "b1": 0.5,
    "b2": 0.999,
    "lambda1": 60,
}

# Save


def save_image(images, path, nrow=8, value_range=(0, 1)):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    vutils.save_image(images, path, nrow, normalize=True, value_range=value_range)


# Compute PSNR between two images
def psnr(imgs1, imgs2):
    mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
    mse += 100 * torch.eq(mse, 0).float()
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


# Load a checkpoint of VAE1 and visualize the results on one batch of either the train or validation set
def visualize_VAE1(dataset="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE1_model = VAE1.load_from_checkpoint(
        "vae1.ckpt", params=params, device=device
    ).to(device)

    data_module = GenericDataModule("datasets", batch_size=32, phase="A")
    data_module.setup()
    if dataset == "train":
        dataloader = data_module.train_dataloader()
    elif dataset == "val":
        dataloader = data_module.val_dataloader()
    else:
        raise Exception("Invalid dataset")
    image, _ = next(iter(dataloader))
    image = image.to(device)
    recons = VAE1_model.vae.decode(VAE1_model.vae.encode(image))
    Path(f"vae1_vis").mkdir(exist_ok=True, parents=True)
    save_image(recons.data, os.path.join("vae1_vis", f"recons.png"))
    save_image(image.data, os.path.join("vae1_vis", f"input.png"))
    np.savetxt("vae1_vis/psnr.txt", psnr(recons, image).detach().cpu().numpy())


# Load a checkpoint of VAE2 and visualize the results on one batch of either the train or validation set
def visualize_VAE2(dataset="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE2_model = VAE2.load_from_checkpoint(
        "vae2.ckpt", params=params, device=device
    ).to(device)

    data_module = GenericDataModule("datasets/non_noisy", batch_size=32, phase="B")
    data_module.setup()
    if dataset == "train":
        dataloader = data_module.train_dataloader()
    elif dataset == "val":
        dataloader = data_module.val_dataloader()
    else:
        raise Exception("Invalid dataset")
    image, _ = next(iter(dataloader))
    image = image.to(device)
    recons = VAE2_model.vae.decode(VAE2_model.vae.encode(image))
    Path(f"vae2_vis").mkdir(exist_ok=True, parents=True)
    save_image(recons.data, os.path.join("vae2_vis", f"recons.png"))
    save_image(image.data, os.path.join("vae2_vis", f"input.png"))
    np.savetxt("vae2_vis/psnr.txt", psnr(recons, image).detach().cpu().numpy())


# Load a checkpoint of the full model and visualize the results on one batch of either the train or validation set
def visualize_full(dataset="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae1_encoder = VAE1.load_from_checkpoint(
        "vae1.ckpt", params=params, device="cpu"
    ).vae.encoder
    vae2 = VAE2.load_from_checkpoint("vae2.ckpt", params=params, device="cpu").vae
    full_model = Mapping.load_from_checkpoint(
        "full.ckpt", vae1_encoder=vae1_encoder, vae2=vae2, params=params, device=device
    ).to(device)
    visualize_full_synth(dataset, full_model)
    visualize_full_real(dataset, full_model)


# Visualize the results of the full model on one batch of images from the synthetic dataset
def visualize_full_synth(dataset="train", full_model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = GenericDataModule("datasets", batch_size=8, phase="Mapping")
    data_module.setup()
    if dataset == "train":
        dataloader = data_module.train_dataloader()
    elif dataset == "val":
        dataloader = data_module.val_dataloader()
    else:
        raise Exception("Invalid dataset")
    imgs, _ = next(iter(dataloader))
    clean_input = imgs[:, 0, :, :, :].to(device)
    noisy_input = imgs[:, 1, :, :, :].to(device)
    denoised, _, _ = full_model(noisy_input)
    Path(f"full_vis/synth").mkdir(exist_ok=True, parents=True)
    save_image(denoised.data, os.path.join("full_vis", "synth", f"denoised.png"))
    save_image(clean_input.data, os.path.join("full_vis", "synth", f"clean_input.png"))
    save_image(noisy_input.data, os.path.join("full_vis", "synth", f"noisy_input.png"))
    np.savetxt("full_vis/psnr.txt", psnr(clean_input, denoised).detach().cpu().numpy())


# Visualize the results of the full model on one batch of images from the real noisy image dataset
def visualize_full_real(dataset="train", full_model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_module = GenericDataModule("datasets/noisy", batch_size=8, phase="Vanilla")
    data_module.setup()
    if dataset == "train":
        dataloader = data_module.train_dataloader()
    elif dataset == "val":
        dataloader = data_module.val_dataloader()
    else:
        raise Exception("Invalid dataset")
    img, _ = next(iter(dataloader))
    img = img.to(device)
    denoised, _, _ = full_model(img)
    Path(f"full_vis/real").mkdir(exist_ok=True, parents=True)
    save_image(denoised.data, os.path.join("full_vis", "real", f"denoised.png"))
    save_image(img.data, os.path.join("full_vis", "real", f"original.png"))


if __name__ == "__main__":
    # visualize_full()
    visualize_VAE1()
    visualize_VAE2()
