from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar


DEFAULT_HPARAMS = {
    "lr": 2e-4,
    "a_reconst": 10,
    "b1": 0.5,
    "b2": 0.999,
    "lambda1_recons": 60,
    "lambda2_feat": 10,
}

DEFAULT_TRAIN_PARAMS = {
    "max_epochs": 1000,
    "gpus": 1,
    "log_every_n_steps": 1,
    "check_val_every_n_epoch": 50,
    "data_dir": "datasets",
    "batch_size": 8,
    "log_dir": None,
    "exp_name": None,
}

# TODO : Allow to resume training
# TODO : Save hyperparameters in the checkpoint
# TODO : Log on epoch instead of step
# TODO : Allow to specify the log directory and experiment name


def train_VAE1(hparams, train_params):
    VAE1_model = VAE1(hparams)

    data_module = GenericDataModule(
        train_params["data_dir"], batch_size=train_params["batch_size"], phase="A"
    )

    trainer = Trainer(
        accelerator="auto",
        devices=train_params["gpus"],
        max_epochs=train_params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=train_params["log_every_n_steps"],
        check_val_every_n_epoch=train_params["check_val_every_n_epoch"],
    )

    trainer.fit(VAE1_model, data_module)


def train_VAE2(hparams, train_params):
    VAE2_model = VAE2(hparams)

    data_module = GenericDataModule(
        train_params["data_dir"], batch_size=train_params["batch_size"], phase="B"
    )

    trainer = Trainer(
        accelerator="auto",
        devices=train_params["gpus"],
        max_epochs=train_params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=train_params["log_every_n_steps"],
        check_val_every_n_epoch=train_params["check_val_every_n_epoch"],
    )

    trainer.fit(VAE2_model, data_module)


def train_Mapping(hparams, train_params, vae1_ckpt_path, vae2_ckpt_path):
    vae1_encoder = VAE1.load_from_checkpoint(vae1_ckpt_path, params=hparams).vae.encoder
    vae2 = VAE2.load_from_checkpoint(vae2_ckpt_path, params=hparams).vae
    mapping_model = Mapping(vae1_encoder, vae2, hparams)

    data_module = GenericDataModule(
        train_params["data_dir"],
        batch_size=train_params["batch_size"],
        phase="Mapping",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=train_params["gpus"],
        max_epochs=train_params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=train_params["log_every_n_steps"],
        check_val_every_n_epoch=train_params["check_val_every_n_epoch"],
    )

    trainer.fit(mapping_model, data_module)


if __name__ == "__main__":
    train_VAE1(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS)
    # train_VAE2(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS)
    # train_Mapping(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS, "vae1.ckpt", "vae2.ckpt")
