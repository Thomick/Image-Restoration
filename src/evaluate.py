# Some utility functions used during developement to evaluate the models and visualize the results.
# Examples are provided at the end of the file.

from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
import torch
from pathlib import Path
import os
import numpy as np
from utils import psnr, save_image, lpips, rescale_colors
from torchvision.transforms import functional as F
from torchvision import io

device = "cpu"


def load_images(dataset="train", image_list=None, phase="A"):
    if image_list == None:
        data_module = GenericDataModule("datasets", batch_size=32, phase="B")
        data_module.setup()
        if dataset == "train":
            dataloader = data_module.train_dataloader()
        elif dataset == "val":
            dataloader = data_module.val_dataloader()
        else:
            raise Exception("Invalid dataset")
        image, _ = next(iter(dataloader))
    else:
        image = torch.stack(
            [
                rescale_colors(
                    F.center_crop(io.read_image(f"datasets/{name}").float(), 256)
                )
                for name in image_list
            ]
        )
    image = image.to(device)
    return image


# Load a checkpoint of VAE1 and visualize the results on one batch of either the train or validation set
def visualize_VAE1(dataset="train", name_list=None, ckpt_path="vae1.ckpt"):
    VAE1_model = VAE1.load_from_checkpoint("vae1.ckpt", device=device).to(device)

    image = load_images(dataset, name_list, phase="A")
    recons = VAE1_model.vae.decode(VAE1_model.vae.encode(image))
    Path(f"vae1_vis").mkdir(exist_ok=True, parents=True)
    save_image(recons.data, os.path.join("vae1_vis", f"recons.png"))
    save_image(image.data, os.path.join("vae1_vis", f"input.png"))
    np.savetxt("vae1_vis/psnr.txt", psnr(recons, image).detach().cpu().numpy())


# Load a checkpoint of VAE2 and visualize the results on one batch of either the train or validation set
def visualize_VAE2(dataset="train", name_list=None, ckpt_path="vae2.ckpt"):
    VAE2_model = VAE2.load_from_checkpoint(ckpt_path, device=device).to(device)

    image = load_images(dataset, name_list, phase="B")

    recons = VAE2_model.vae.decode(VAE2_model.vae.encode(image))
    Path(f"vae2_vis").mkdir(exist_ok=True, parents=True)
    save_image(recons.data, os.path.join("vae2_vis", f"recons.png"))
    save_image(image.data, os.path.join("vae2_vis", f"input.png"))
    np.save("vae2_vis/test.npy", recons[0].detach().cpu().numpy())
    psnr_metric = psnr(recons, image).detach().cpu().numpy()
    np.savetxt("vae2_vis/psnr.txt", psnr_metric)
    lpips_metric = lpips(recons, image, device).detach().cpu().numpy()
    np.savetxt("vae2_vis/lpips.txt", lpips_metric)
    print("psnr:" + str(np.mean(psnr_metric)))
    print("lpips:" + str(np.mean(lpips_metric)))


def evaluate_VAE2_fulldataset(dataset="train", ckpt_path="vae2.ckpt"):

    VAE2_model = VAE2.load_from_checkpoint(ckpt_path, device=device).to(device)

    print(VAE2_model)

    data_module = GenericDataModule("datasets", batch_size=32, phase="B")
    data_module.setup()
    if dataset == "train":
        dataloader = data_module.train_dataloader()
    elif dataset == "val":
        dataloader = data_module.val_dataloader()
    else:
        raise Exception("Invalid dataset")
    psnr_metric = []
    lpips_metric = []
    for image, _ in iter(dataloader):
        image = image.to(device)
        # recons = VAE2_model.vae.decode(VAE2_model.vae.encode(image))
        recons = VAE2_model(image)
        psnr_metric += psnr(recons, image).detach().cpu().numpy().tolist()
        lpips_metric += lpips(recons, image, device).detach().cpu().numpy().tolist()

    print(f"average psnr({dataset}) : {np.mean(psnr_metric)}")
    print(f"average lpips({dataset}) : {np.mean(lpips_metric)}")


# Load a checkpoint of the full model and visualize the results on one batch of either the train or validation set
def visualize_full(dataset="train", ckpt_path="full.ckpt"):
    full_model = Mapping.load_from_checkpoint(
        ckpt_path,
        device=device,
    ).to(device)
    visualize_full_synth(dataset, full_model)
    visualize_full_real(dataset, full_model)


# Visualize the results of the full model on one batch of images from the synthetic dataset
def visualize_full_synth(dataset="train", full_model=None):
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


def test_reconstruction(img_list, vae2_path):
    with torch.no_grad():
        vae = VAE2.load_from_checkpoint(
            vae2_path,
            strict=False,
        ).vae.to(device)

        for img_path in img_list:
            img = rescale_colors(io.read_image(f"datasets/test/{img_path}").float())
            if img.shape[0] == 4:
                img = img[(0, 1, 2), :, :].unsqueeze(0).to(device)
            else:
                img = img.unsqueeze(0).to(device)
            print(img.shape)
            recons = vae.encode_decode(img).detach()
            save_image(
                recons.data,
                os.path.join("test_vae2_vis", f"{img_path.split('.')[0]}.png"),
            )
            del img
            del recons
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # visualize_full(dataset="val", ckpt_path="full.ckpt")
    # visualize_VAE1(dataset="val", ckpt_path="vae1.ckpt")
    test_images = [
        "Flickr500/Img500.png",
        "Flickr500/Img499.png",
        "Flickr500/Img498.png",
        "Flickr500/Img497.png",
        "Flickr500/Img496.png",
        "Flickr500/Img495.png",
        "Flickr500/Img494.png",
        "Flickr500/Img493.png",
    ]
    # visualize_VAE2(dataset="val", name_list=test_images, ckpt_path="vae2.ckpt")

    # evaluate_VAE2_fulldataset("train", "vae2nodeconv.ckpt")
    # evaluate_VAE2_fulldataset("val", "vae2nodeconv.ckpt")

    test_images = [
        "Img492.png",
        "Img491.png",
        "Img490.png",
        "Img489.png",
        "Img488.png",
        "Img487.png",
        "Img486.png",
        "Img485.png",
    ]
    # test_reconstruction(test_images, "checkpoints/vae2originalpascal.ckpt")
