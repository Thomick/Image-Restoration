from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


data_transform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # TODO : Check which normalization should be used
])

# Simple dataset class for loading images from a folder without labels
# TODO: Rename the class to something more appropriate


class MyDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str):
        # TODO :  Add a transform parameter
        self.data_dir = Path(data_dir)
        imgs = sorted(
            [f for f in self.data_dir.iterdir() if f.suffix == '.png'])

        self.imgs = imgs[:int(
            len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        img = data_transform(img)

        return img, 0.0


# DataModule for the VAE1 model
# TODO: Rename the class to something more appropriate
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
