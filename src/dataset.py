from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms, io
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
from utils import rescale_colors


def randomJPEGcompression(image):
    qf = np.random.randint(40, 100)
    res = io.decode_jpeg(io.encode_jpeg(image.type(torch.uint8), qf)).type_as(image)
    return res


def randomGaussianBlur(image):
    sigma = np.random.uniform(1.0, 5.0)
    k = np.random.choice([3, 5, 7])
    return torch.clamp(TF.gaussian_blur(image, (k, k), sigma), 0.0, 255.0)


def randomGaussianNoise(image):
    sigma = np.random.uniform(5.0, 50.0)
    return torch.clamp(image + torch.randn_like(image) * sigma, 0.0, 255.0)


def randomColorJitter(image):
    # A bit strange, but it is used in the paper
    delta = np.random.uniform(-20.0, 20.0)
    return torch.clamp(image + delta, 0.0, 255.0)


data_transform = transforms.Compose(
    [
        transforms.RandomCrop(256, pad_if_needed=True),
        transforms.RandomHorizontalFlip(0.5),
    ]
)

random_degradation = transforms.RandomOrder(
    [
        transforms.RandomApply([transforms.Lambda(randomGaussianNoise)], p=0.7),
        transforms.RandomApply([transforms.Lambda(randomGaussianBlur)], p=0.7),
        transforms.RandomApply([transforms.Lambda(randomJPEGcompression)], p=0.7),
        transforms.RandomApply([transforms.Lambda(randomColorJitter)], p=0.7),
    ]
)


class VanillaDataset(Dataset):
    def __init__(self, data_dir: str, split: str, split_ratio=0.75):
        # load images
        self.data_dir = Path(data_dir)
        imgs = sorted(
            [
                str(f)
                for f in self.data_dir.iterdir()
                if f.suffix == ".png" or f.suffix == ".jpg"
            ]
        )

        self.imgs = (
            imgs[: int(len(imgs) * split_ratio)]
            if split == "train"
            else imgs[int(len(imgs) * split_ratio) :]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = io.read_image(self.imgs[idx]).float()
        img = data_transform(img)

        return rescale_colors(img), 0.0


class MappingDataset(Dataset):
    def __init__(self, data_dir: str, split: str, split_ratio=0.75):
        # Load images
        self.data_dir = Path(data_dir + "/non_noisy")
        imgs = sorted([str(f) for f in self.data_dir.iterdir() if f.suffix == ".png"])

        self.imgs = (
            imgs[: int(len(imgs) * split_ratio)]
            if split == "train"
            else imgs[int(len(imgs) * split_ratio) :]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = io.read_image(self.imgs[idx]).float()
        img = data_transform(img)
        noisy_img = random_degradation(img)

        return torch.stack([rescale_colors(img), rescale_colors(noisy_img)]), 0.0


class PhaseADataset(Dataset):
    def __init__(self, data_dir: str, split: str, split_ratio=0.75):

        # Load noisy images
        real_data_dir = Path(data_dir + "/noisy")
        real_imgs = sorted(
            [str(f) for f in real_data_dir.iterdir() if f.suffix == ".png"]
        )

        real_imgs = (
            real_imgs[: int(len(real_imgs) * split_ratio)]
            if split == "train"
            else real_imgs[int(len(real_imgs) * split_ratio) :]
        )

        # Load clean images to which we will add synthetic noise
        data_for_synthesis_dir = Path(data_dir + "/non_noisy")
        imgs_for_synthesis = sorted(
            [str(f) for f in data_for_synthesis_dir.iterdir() if f.suffix == ".png"]
        )
        imgs_for_synthesis = (
            imgs_for_synthesis[: int(len(imgs_for_synthesis) * split_ratio)]
            if split == "train"
            else imgs_for_synthesis[int(len(imgs_for_synthesis) * split_ratio) :]
        )

        # Combine the two sets of images
        self.imgs = [(path, 1.0) for path in real_imgs] + [
            (path, 0.0) for path in imgs_for_synthesis
        ]
        np.random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = io.read_image(self.imgs[idx][0]).float()
        if self.imgs[idx][1] == 0.0:  # if image is clean then we add synthetic noise
            img = random_degradation(img)
        img = data_transform(img)

        return rescale_colors(img), self.imgs[idx][1]


# DataModule for the 3 phases : A, B, Mapping
class GenericDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        phase: str = "A",
        split_ratio=0.75,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase
        self.split_ratio = split_ratio

    def setup(self, stage=None) -> None:
        if self.phase == "A":
            self.val_dataset = PhaseADataset(
                self.data_dir, split="val", split_ratio=self.split_ratio
            )
            self.train_dataset = PhaseADataset(
                self.data_dir, split="train", split_ratio=self.split_ratio
            )
        elif self.phase == "B":
            self.val_dataset = VanillaDataset(
                self.data_dir + "/non_noisy", split="val", split_ratio=self.split_ratio
            )
            self.train_dataset = VanillaDataset(
                self.data_dir + "/non_noisy",
                split="train",
                split_ratio=self.split_ratio,
            )
        elif self.phase == "Mapping":
            self.val_dataset = MappingDataset(
                self.data_dir, split="val", split_ratio=self.split_ratio
            )
            self.train_dataset = MappingDataset(
                self.data_dir, split="train", split_ratio=self.split_ratio
            )
        elif self.phase == "Vanilla":
            self.val_dataset = VanillaDataset(
                self.data_dir, split="val", split_ratio=self.split_ratio
            )
            self.train_dataset = VanillaDataset(
                self.data_dir, split="train", split_ratio=self.split_ratio
            )
        else:
            raise Exception("Invalid phase")
        print("Train dataset size : ", len(self.train_dataset))

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
