import argparse

# TODO : Use more explicit names for the arguments
# TODO : Add support for config files

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


class CliParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Parse train params
        self.parser.add_argument(
            "--max_epochs",
            required=True,
            type=int,
            default=DEFAULT_TRAIN_PARAMS["max_epochs"],
        )
        self.parser.add_argument(
            "--gpus", type=int, default=DEFAULT_TRAIN_PARAMS["gpus"]
        )
        self.parser.add_argument(
            "--log_every_n_steps",
            type=int,
            default=DEFAULT_TRAIN_PARAMS["log_every_n_steps"],
        )
        self.parser.add_argument(
            "--check_val_every_n_epoch",
            type=int,
            default=DEFAULT_TRAIN_PARAMS["check_val_every_n_epoch"],
        )
        self.parser.add_argument(
            "--sample_rate",
            type=int,
            default=DEFAULT_TRAIN_PARAMS["sample_images_every_n_epoch"],
        )
        self.parser.add_argument(
            "--data_dir",
            required=True,
            type=str,
            default=DEFAULT_TRAIN_PARAMS["data_dir"],
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=DEFAULT_TRAIN_PARAMS["batch_size"]
        )
        self.parser.add_argument(
            "--use_perceptual_loss",
            type=bool,
            default=DEFAULT_TRAIN_PARAMS["use_perceptual_loss"],
        )

        # Parse hyperparams
        self.parser.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
        self.parser.add_argument(
            "--a_reconst", type=float, default=DEFAULT_HPARAMS["a_reconst"]
        )
        self.parser.add_argument("--b1", type=float, default=DEFAULT_HPARAMS["b1"])
        self.parser.add_argument("--b2", type=float, default=DEFAULT_HPARAMS["b2"])
        self.parser.add_argument(
            "--lambda1_recons", type=float, default=DEFAULT_HPARAMS["lambda1_recons"]
        )
        self.parser.add_argument(
            "--lambda2_feat", type=float, default=DEFAULT_HPARAMS["lambda2_feat"]
        )
        self.parser.add_argument(
            "--use_transpose_conv",
            type=bool,
            default=DEFAULT_HPARAMS["use_transpose_conv"],
        )
        self.parser.add_argument(
            "--interp_mode", type=str, default=DEFAULT_HPARAMS["interp_mode"]
        )
        self.parser.add_argument(
            "--upsampling_kernel_size",
            type=int,
            default=DEFAULT_HPARAMS["upsampling_kernel_size"],
        )

    def parse(self):
        opt = self.parser.parse_args()
        train_params = {
            "max_epochs": opt.max_epochs,
            "gpus": opt.gpus,
            "log_every_n_steps": opt.log_every_n_steps,
            "check_val_every_n_epoch": opt.check_val_every_n_epoch,
            "sample_images_every_n_epoch": opt.sample_rate,
            "data_dir": opt.data_dir,
            "batch_size": opt.batch_size,
            "use_perceptual_loss": opt.use_perceptual_loss,
        }
        hparams = {
            "lr": opt.lr,
            "a_reconst": opt.a_reconst,
            "b1": opt.b1,
            "b2": opt.b2,
            "lambda1_recons": opt.lambda1_recons,
            "lambda2_feat": opt.lambda2_feat,
            "use_transpose_conv": opt.use_transpose_conv,
            "interp_mode": opt.interp_mode,
            "upsampling_kernel_size": opt.upsampling_kernel_size,
        }
        return train_params, hparams
