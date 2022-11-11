import torch
from pathlib import Path
import torchvision.utils as vutils

DEFAULT_VALUE_RANGE = (-1, 1)

# Save a batch of images
def save_image(images, path, nrow=8, value_range=DEFAULT_VALUE_RANGE):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    vutils.save_image(images, path, nrow=nrow, normalize=True, value_range=value_range)


# Compute PSNR between two images
def psnr(imgs1, imgs2, value_range=DEFAULT_VALUE_RANGE):
    mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
    PIXEL_MAX = value_range[1] - value_range[0]
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
