from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
import torch
from pathlib import Path
import os
import numpy as np
from utils import psnr, save_image, lpips, rescale_colors
from torchvision.transforms import functional as F
from torchvision import io
from train import DEFAULT_HPARAMS
import tqdm

device = "cpu"

# TODO: Remove params from the checkpoint loading (try two step initialization)

# Load a checkpoint of VAE1 and visualize the results on one batch of either the train or validation set
def visualize_VAE1(dataset="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE1_model = VAE1.load_from_checkpoint(
        "vae1.ckpt", params=DEFAULT_HPARAMS, device=device
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
def visualize_VAE2(dataset="train", name_list=None, ckpt_path="vae2.ckpt"):

    VAE2_model = VAE2.load_from_checkpoint(
        ckpt_path, params=DEFAULT_HPARAMS, device=device
    ).to(device)

    print(VAE2_model)

    if name_list == None:
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
                for name in name_list
            ]
        )
    image = image.to(device)
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

    VAE2_model = VAE2.load_from_checkpoint(
        ckpt_path, params=DEFAULT_HPARAMS, device=device
    ).to(device)

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
def visualize_full(dataset="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae1_encoder = VAE1.load_from_checkpoint(
        "vae1.ckpt", params=DEFAULT_HPARAMS, device="cpu"
    ).vae.encoder
    vae2 = VAE2.load_from_checkpoint(
        "vae2.ckpt", params=DEFAULT_HPARAMS, device="cpu"
    ).vae
    full_model = Mapping.load_from_checkpoint(
        "full.ckpt",
        params=DEFAULT_HPARAMS,
        vae1_encoder=vae1_encoder,
        vae2=vae2,
        device=device,
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


def test_denoising(img_list, mapping_path, vae1_path, vae2_path):
    with torch.no_grad():
        full_model = Mapping.load_from_checkpoint(
            mapping_path,
            params=DEFAULT_HPARAMS,
            vae1_ckpt_path=vae1_path,
            vae2_ckpt_path=vae2_path,
            device=device,
        ).to(device)

        for img_path in img_list:
            img = rescale_colors(io.read_image(f"datasets/test/{img_path}").float())
            if img.shape[0] == 4:
                img = img[(0, 1, 2), :, :].unsqueeze(0).to(device)
            else:
                img = img.unsqueeze(0).to(device)
            print(img.shape)
            denoised, _, _ = full_model(img)
            save_image(
                denoised.data,
                os.path.join("test_denoising_vis", f"{img_path.split('.')[0]}.png"),
            )
            del img
            del denoised
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_images = [
        "Flickr500/Img500.png",
        "Flickr500/Img499.png",
        "Flickr500/Img498.png",
        "Flickr500/Img497.png",
        "Flickr500/Img496.png",
        "Flickr500/Img495.png",
        "Flickr500/Img494.png",
        "Flickr500/Img493.png",
        "Flickr500/Img492.png",
        "Flickr500/Img491.png",
        "Flickr500/Img490.png",
        "Flickr500/Img489.png",
        "Flickr500/Img488.png",
        "Flickr500/Img487.png",
        "Flickr500/Img486.png",
        "Flickr500/Img485.png",
    ]
    # visualize_full(dataset="val")
    # visualize_VAE1(dataset="val")
    # DEFAULT_HPARAMS["use_transpose_conv"] = True
    """ visualize_VAE2(
        dataset="val", name_list=test_images, ckpt_path="vae2nodeconvpascal.ckpt"
    ) """
    # evaluate_VAE2_fulldataset("train", "vae2nodeconv.ckpt")
    # evaluate_VAE2_fulldataset("val", "vae2nodeconv.ckpt")

    test_images = [
        # "a.png",
        # "b.png",
        # "c.png",
        # "d.png",
        # "e.png",
        # "f.png",
        # "g.png",
        # "h.png",
        "241.jpg",
        "389.png",
        "370.png",
    ]
    test_denoising(
        test_images, "fullnodeconv.ckpt", "vae1nodeconv.ckpt", "vae2nodeconv.ckpt"
    )
