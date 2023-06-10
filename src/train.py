from dataset import GenericDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import argparse
from config import load_training_parameters


def train_VAE1(params, output_path=None, checkpoint_path=None):
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
    from models import VAE1

    VAE1_model = VAE1(params)
    print(VAE1_model)

    data_module = GenericDataModule(
        params["data_dir"], batch_size=params["batch_size"], phase="A"
    )

    trainer = Trainer(
        accelerator="auto",
        devices=params["gpus"],
        max_epochs=params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=params["log_every_n_steps"],
        check_val_every_n_epoch=params["check_val_every_n_epoch"],
    )

    trainer.fit(VAE1_model, data_module, ckpt_path=checkpoint_path)
    trainer.save_checkpoint(output_path)


def train_VAE2(params, output_path=None, checkpoint_path=None):
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
    from models import VAE2

    VAE2_model = VAE2(params)
    print(VAE2_model)

    data_module = GenericDataModule(
        params["data_dir"], batch_size=params["batch_size"], phase="B"
    )

    trainer = Trainer(
        accelerator="auto",
        devices=params["gpus"],
        max_epochs=params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=params["log_every_n_steps"],
        check_val_every_n_epoch=params["check_val_every_n_epoch"],
    )

    trainer.fit(VAE2_model, data_module, ckpt_path=checkpoint_path)
    trainer.save_checkpoint(output_path)


def train_Mapping(
    params,
    output_path=None,
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
    from models import Mapping

    mapping_model = Mapping(params, params["vae1_ckpt_path"], params["vae2_ckpt_path"])
    print(mapping_model)

    data_module = GenericDataModule(
        params["data_dir"],
        batch_size=params["batch_size"],
        phase="Mapping",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=params["gpus"],
        max_epochs=params["max_epochs"],
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=params["log_every_n_steps"],
        check_val_every_n_epoch=params["check_val_every_n_epoch"],
    )

    trainer.fit(mapping_model, data_module, ckpt_path=checkpoint_path)
    trainer.save_checkpoint(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "-c", "--cfg-path", required=True, help="path to configuration file."
    )
    parser.add_argument(
        "-s",
        "--stage",
        required=True,
        help="stage of training. Can be vae1, vae2 or mapping.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="path to save the trained model.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=False,
        help="path to checkpoint file. Allows to resume training from a checkpoint.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    training_params = load_training_parameters(args.cfg_path)

    if args.stage == "vae1":
        train_VAE1(training_params, args.output_path, args.checkpoint_path)
    elif args.stage == "vae2":
        train_VAE2(training_params, args.output_path, args.checkpoint_path)
    elif args.stage == "mapping":
        train_Mapping(
            training_params,
            args.output_path,
            args.checkpoint_path,
        )

    pass
