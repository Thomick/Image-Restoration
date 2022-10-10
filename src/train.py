from ast import Raise
from dataset import VAEDataModule
from models import VAE2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch

params = {
    'lr': 2e-4,
    'a_reconst': 10,
    'b1': 0.5,
    'b2': 0.999,
}


def train_VAE1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE1_model = VAE1(params).to(device)

    raise Exception("VAE1DataModule not implemented")
    # TODO: Implement VAE1DataModule
    data_module = None

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=3,
                      callbacks=[TQDMProgressBar(refresh_rate=20)],
                      log_every_n_steps=10)

    trainer.fit(VAE1_model, data_module)


def train_VAE2():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE2_model = VAE2(params).to(device)

    data_module = VAEDataModule("datasets/Flickr500")

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=100,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=1)

    trainer.fit(VAE2_model, data_module)
