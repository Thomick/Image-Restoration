from dataset import GenericDataModule
from models import VAE2, VAE1, Mapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# TODO : Remove hardcoded paths
# TODO : Add command line arguments

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
    "max_epochs": 300,
    "gpus": 1,
    "log_every_n_steps": 10,
    "check_val_every_n_epoch": 1,
    "sample_images_every_n_epoch": 10,
    "data_dir": "datasets",
    "batch_size": 16,
    "use_perceptual_loss": True,
}


def train_VAE1(
    hparams=DEFAULT_HPARAMS, train_params=DEFAULT_TRAIN_PARAMS, checkpoint_path=None
):
    """
    Train VAE1 model. Periodically saves the trained model to a checkpoint file

    Parameters
    ----------
    hparams : dict, optional
        Hyperparameters, by default DEFAULT_HPARAMS
    train_params : dict, optional
        Training parameters, by default DEFAULT_TRAIN_PARAMS
    checkpoint_path : str, optional
        Path to checkpoint, by default None. If None, train from scratch. If a path is provided, the model will be loaded from the checkpoint and resume training but a new checkpoint path will be created
    """
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
    print(f"Checkpoint path: {trainer.ckpt_path}")

    trainer.fit(VAE1_model, data_module, ckpt_path=checkpoint_path)


def train_VAE2(
    hparams=DEFAULT_HPARAMS, train_params=DEFAULT_TRAIN_PARAMS, checkpoint_path=None
):
    """
    Train VAE2 model. Periodically saves the trained model to a checkpoint file

    Parameters
    ----------
    hparams : dict, optional
        Hyperparameters, by default DEFAULT_HPARAMS
    train_params : dict, optional
        Training parameters, by default DEFAULT_TRAIN_PARAMS
    checkpoint_path : str, optional
        Path to checkpoint, by default None. If None, train from scratch. If a path is provided, the model will be loaded from the checkpoint and resume training but a new checkpoint path will be created
    """
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
    print(f"Checkpoint path: {trainer.ckpt_path}")

    trainer.fit(VAE2_model, data_module, ckpt_path=checkpoint_path)


def train_Mapping(
    hparams=DEFAULT_HPARAMS,
    train_params=DEFAULT_TRAIN_PARAMS,
    vae1_ckpt_path="vae1.ckpt",
    vae2_ckpt_path="vae2.ckpt",
    checkpoint_path=None,
):
    """
    Train mapping model. Periodically saves the trained model to a checkpoint file

    Parameters
    ----------
    hparams : dict, optional
        Hyperparameters, by default DEFAULT_HPARAMS
    train_params : dict, optional
        Training parameters, by default DEFAULT_TRAIN_PARAMS
    checkpoint_path : str, optional
        Path to checkpoint, by default None. If None, train from scratch. If a path is provided, the model will be loaded from the checkpoint and resume training but a new checkpoint path will be created
    """
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
    print(f"Checkpoint path: {trainer.ckpt_path}")

    trainer.fit(mapping_model, data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":

    # Set to None to train from scratch
    checkpoint_path = None

    # train_VAE2(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    # train_VAE1(DEFAULT_HPARAMS, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    hparams = DEFAULT_HPARAMS
    hparams["use_transpose_conv"] = False
    # train_VAE1(hparams, DEFAULT_TRAIN_PARAMS, checkpoint_path=checkpoint_path)
    train_params = DEFAULT_TRAIN_PARAMS
    train_params["batch_size"] = 7
    train_params["max_epochs"] = 100
    train_Mapping(
        hparams,
        train_params,
        "vae1nodeconvpascal.ckpt",
        "vae2nodeconvpascal.ckpt",
        checkpoint_path=checkpoint_path,
    )
    pass
