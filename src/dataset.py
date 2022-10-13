from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, io, datasets
from pathlib import Path
import torch
import numpy as np
from PIL import Image


data_transform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True),
    transforms.RandomHorizontalFlip(0.5),
])


class MyDataset(Dataset):
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
        img = 2*io.read_image(self.imgs[idx])/255-1
        img = data_transform(img)

        return img, 0.0


class PhaseADataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str):
        # TODO :  Add a transform parameter
        self.real_data_dir = Path(data_dir+"/noisy")
        real_imgs = sorted(
            [str(f) for f in self.real_data_dir.iterdir() if f.suffix == '.png'])

        self.real_imgs = real_imgs[:int(
            len(real_imgs) * 0.75)] if split == "train" else real_imgs[int(len(real_imgs) * 0.75):]

        self.data_for_synthesis_dir = Path(data_dir+"/non_noisy")
        imgs_for_synthesis = sorted(
            [str(f) for f in self.data_for_synthesis_dir.iterdir() if f.suffix == '.png'])
        self.imgs_for_synthesis = imgs_for_synthesis[:int(len(
            imgs_for_synthesis) * 0.75)] if split == "train" else imgs_for_synthesis[int(len(imgs_for_synthesis) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # TODO : Add images with synthetic noise
        img = 2*io.read_image(self.real_imgs[idx])/255-1
        img = data_transform(img)

        return img, 0.0


class VAEDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:

        self.train_dataset = MyDataset(self.data_dir, split="train")

        # TODO: Separate training and validation set
        self.val_dataset = MyDataset(self.data_dir, split="val")

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
