from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
from pathlib import Path


hparams = {
    'lr': 2e-4,
    'a_reconst': 10,
    'b1': 0.5,
    'b2': 0.999,
    'lambda1_recons': 60,
    'lambda2_feat': 10,
}

# TODO : Add parameters to specify the training parameters (dataset, batch size, etc.)
# TODO : Allow to resume training


def train_VAE1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE1_model = VAE1(hparams).to(device)

    data_module = GenericDataModule("datasets", batch_size=16, phase="A")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VAE2_model = VAE2(hparams).to(device)

    data_module = GenericDataModule(
        "datasets", batch_size=16, phase="B")

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=1000,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=10,
                      check_val_every_n_epoch=50)

    Path(f"{trainer.log_dir}/Input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer.fit(VAE2_model, data_module)


def train_Mapping():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae1_chkp_path = "vae1.ckpt"
    vae2_chkp_path = "vae2.ckpt"
    vae1_encoder = VAE1.load_from_checkpoint(
        vae1_chkp_path, params=hparams, device='cpu').vae.encoder
    vae2 = VAE2.load_from_checkpoint(
        vae2_chkp_path, params=hparams, device='cpu').vae
    mapping_model = Mapping(vae1_encoder, vae2, hparams).to(device)

    data_module = GenericDataModule("datasets", batch_size=8, phase="Mapping")

    trainer = Trainer(accelerator=device,
                      devices=1 if device == "cuda" else None,
                      max_epochs=1000,
                      callbacks=[TQDMProgressBar(refresh_rate=1)],
                      log_every_n_steps=10,
                      check_val_every_n_epoch=5)

    Path(f"{trainer.log_dir}/Clean_input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Noisy_input").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.log_dir}/Results").mkdir(exist_ok=True, parents=True)

    trainer.fit(mapping_model, data_module)
