from dataset import VAEDataModule
from models import VAE2, VAE1
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

    VAE1_model = VAE1(params, device).to(device)

    data_module = VAEDataModule("datasets", batch_size=32, phase="A")

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=1000,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=10,
                      check_val_every_n_epoch=50)

    Path(f"{trainer.log_dir}/Input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer.fit(VAE1_model, data_module)


def train_VAE2():
    # TODO : Allow to resume training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE2_model = VAE2(params).to(device)

    data_module = VAEDataModule("datasets/non_noisy", batch_size=32, phase="B")

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=1000,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=10,
                      check_val_every_n_epoch=50)

    Path(f"{trainer.log_dir}/Input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer.fit(VAE2_model, data_module)
