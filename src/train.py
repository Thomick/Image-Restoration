from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# Set to None to train from scratch
checkpoint_path = None

DEFAULT_HPARAMS = {
    "lr": 2e-4,
    "a_reconst": 10,
    "b1": 0.5,
    "b2": 0.999,
    "lambda1_recons": 60,
    "lambda2_feat": 10,
    "use_transpose_conv": False,
    "interp_mode": "nearest",
    "upsampling_kernel_size": 5,
}

DEFAULT_TRAIN_PARAMS = {
    "max_epochs": 200,
    "gpus": 1,
    "log_every_n_steps": 10,
    "check_val_every_n_epoch": 1,
    "sample_images_every_n_epoch": 10,
    "data_dir": "datasets",
    "batch_size": 16,
    "use_perceptual_loss": True,
}


# TODO : Allow to resume training


def train_VAE1(
    hparams=DEFAULT_HPARAMS, train_params=DEFAULT_TRAIN_PARAMS, checkpoint_path=None
):
    params = {**hparams, **train_params}
    VAE1_model = VAE1(params)
    print(VAE1_model)

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

    trainer.fit(VAE1_model, data_module, ckpt_path=checkpoint_path)


def train_VAE2(
    hparams=DEFAULT_HPARAMS, train_params=DEFAULT_TRAIN_PARAMS, checkpoint_path=None
):
    params = {**hparams, **train_params}
    VAE2_model = VAE2(params)
    print(VAE2_model)

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

    trainer.fit(VAE2_model, data_module, ckpt_path=checkpoint_path)


def train_Mapping(
    hparams=DEFAULT_HPARAMS,
    train_params=DEFAULT_TRAIN_PARAMS,
    vae1_ckpt_path="vae1.ckpt",
    vae2_ckpt_path="vae2.ckpt",
):
    params = {**hparams, **train_params}
    mapping_model = Mapping(params, vae1_ckpt_path, vae2_ckpt_path)
    print(mapping_model)

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
    # train_VAE2(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    # train_VAE1(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    hparams = DEFAULT_HPARAMS
    hparams["use_transpose_conv"] = True
    train_VAE2(hparams, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    """ train_params = DEFAULT_TRAIN_PARAMS
    train_params["batch_size"] = 7
    train_Mapping(
        DEFAULT_HPARAMS, train_params, "vae1nodeconv.ckpt", "vae2nodeconv.ckpt"
    ) """
    pass
