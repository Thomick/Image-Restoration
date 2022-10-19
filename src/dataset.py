from pickletools import uint8
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms, io
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np


def randomJPEGcompression(image):
    qf = np.random.randint(40, 100)
    res = io.decode_jpeg(io.encode_jpeg(
        image.type(torch.uint8), qf)).type_as(image)
    return res


def randomGaussianBlur(image):
    sigma = np.random.uniform(1.0, 5.0)
    k = np.random.choice([3, 5, 7])
    return torch.clamp(TF.gaussian_blur(image, (k, k), sigma), 0., 255.)


def randomGaussianNoise(image):
    sigma = np.random.uniform(5.0, 50.0)
    return torch.clamp(image+torch.randn_like(image)*sigma, 0., 255.)


def randomColorJitter(image):
    # A bit strange, but it is used in the paper
    delta = np.random.uniform(-20., 20.)

    return torch.clamp(image+delta, 0., 255.)


data_transform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True),
    transforms.RandomHorizontalFlip(0.5),
])

random_degradation = transforms.RandomOrder([
    transforms.RandomApply([transforms.Lambda(randomGaussianNoise)], p=0.7),
    transforms.RandomApply([transforms.Lambda(randomGaussianBlur)], p=0.7),
    transforms.RandomApply([transforms.Lambda(randomJPEGcompression)], p=0.7),
    transforms.RandomApply([transforms.Lambda(randomColorJitter)], p=0.7)])


# Simple dataset class for loading images from a folder without labels
# TODO: Rename the class to something more appropriate


class PhaseBDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str):
        # TODO :  Add a transform parameter
        self.data_dir = Path(data_dir)
        imgs = sorted(
            [str(f) for f in self.data_dir.iterdir() if f.suffix == '.png'])

        self.imgs = imgs[:int(
            len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = io.read_image(self.imgs[idx]).float()
        img = data_transform(img)

        return 2*img/255-1, 0.0


class PhaseADataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str):
        # TODO :  Add a transform parameter
        real_data_dir = Path(data_dir+"/noisy")
        real_imgs = sorted(
            [str(f) for f in real_data_dir.iterdir() if f.suffix == '.png'])

        real_imgs = real_imgs[:int(
            len(real_imgs) * 0.75)] if split == "train" else real_imgs[int(len(real_imgs) * 0.75):]

        data_for_synthesis_dir = Path(data_dir+"/non_noisy")
        imgs_for_synthesis = sorted(
            [str(f) for f in data_for_synthesis_dir.iterdir() if f.suffix == '.png'])
        imgs_for_synthesis = imgs_for_synthesis[:int(len(
            imgs_for_synthesis) * 0.75)] if split == "train" else imgs_for_synthesis[int(len(imgs_for_synthesis) * 0.75):]

        self.imgs = [(path, 1.) for path in real_imgs] + [(path, 0.)
                                                          for path in imgs_for_synthesis]
        np.random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # TODO : Add images with synthetic noise
        img = io.read_image(self.imgs[idx][0]).float()
        if self.imgs[idx][1] == 0.:
            img = random_degradation(img)
        img = data_transform(img)

        return 2*img/255-1, self.imgs[idx][1]


# DataModule for the VAE1 model
# TODO: Rename the class to something more appropriate
class VAEDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        phase: str = "A"
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase

    def setup(self, stage=None) -> None:
        if self.phase == "A":
            self.val_dataset = PhaseADataset(self.data_dir, split="val")
            self.train_dataset = PhaseADataset(self.data_dir, split="train")
        else:
            self.val_dataset = PhaseBDataset(self.data_dir, split="val")
            self.train_dataset = PhaseBDataset(self.data_dir, split="train")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=True,
        )
