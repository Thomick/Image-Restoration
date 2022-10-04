from pytorch_lightning import LightningDataModule
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class MyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str):
        self.data_dir = Path(data_path) / "OxfordPets"
        imgs = sorted(
            [f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(
            len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        return img, 0.0


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self) -> None:

        self.train_dataset = MyDataset(
            self.data_dir,
            split='train',
        )

        self.val_dataset = MyDataset(
            self.data_dir,
            split='test',
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
