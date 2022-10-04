import torch
from pytorch_lightning import Trainer
from models import VAE1
from dataset import VAEDataset
from networks import VAE

params = {
    'LR': 1e-3,
    'weight_decay': 1e-5,
    'kld_weight': 0.01,
}


def train_VAE1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VAE().to(device)
    experiment = VAE1(model, params)

    data = VAEDataset()  # TODO: Define params for the dataset

    runner = Trainer()  # TODO: Define params for the trainer

    runner.fit(experiment, datamodule=data)
