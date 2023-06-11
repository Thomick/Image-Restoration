# Utility functions for image processing and evaluation

import cv2
import os
import torch
from pathlib import Path
import torchvision.utils as vutils
import lpips as lpips_module

DEFAULT_VALUE_RANGE = (-1, 1)

# Rescale colors from [0, 255] to [-1, 1]
def rescale_colors(image):
    return 2 * image / 255.0 - 1.0


# Inverse of rescale_colors
def rescale_colors_back(image):
    return (image + 1.0) * 127.5


# Save a batch of images
def save_image(images, path, nrow=8):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    images = images.clamp(-1, 1)
    images = rescale_colors_back(images) / 255.0
    vutils.save_image(images, path, nrow=nrow, normalize=False)


# Compute PSNR between two images
def psnr(imgs1, imgs2, value_range=DEFAULT_VALUE_RANGE):
    mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
    PIXEL_MAX = value_range[1] - value_range[0]
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


lpips_model = None


def lpips(imgs1, imgs2, device="cuda"):
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips_module.LPIPS(net="vgg", verbose=False)
    lpips_model.to(device)
    return lpips_model.forward(imgs1, imgs2).squeeze()


# Resize all the images of a directory to 1/ratio of the original size and append "DS"(downscaled) to the filenames.
def resize_folder(folder_path, ratio=3.5):
    img_dir = folder_path
    img_list = os.listdir(img_dir)

    for img_path in img_list:
        print(img_path)
        img = cv2.imread(img_dir + img_path, cv2.IMREAD_UNCHANGED)
        width = int(img.shape[1] / ratio)
        height = int(img.shape[0] / ratio)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_dir + img_path[:-4] + "DS.png", img)
