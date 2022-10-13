from logging.config import valid_ident
from dataset import VAEDataModule
from models import VAE2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
from pathlib import Path


params = {
    'lr': 2e-3,
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

    data_module = VAEDataModule("datasets/Flickr500", batch_size=32)

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=1000,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=8,
                      check_val_every_n_epoch=20)

    Path(f"{trainer.log_dir}/Input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer.fit(VAE2_model, data_module)
