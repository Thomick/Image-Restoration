import yaml

DEFAULT_PARAMS = {
    "lr": 2e-4,
    "a_reconst": 10.0,
    "b1": 0.5,
    "b2": 0.999,
    "lambda1_recons": 60.0,
    "lambda2_feat": 10.0,
    "use_transpose_conv": False,
    "interp_mode": "nearest",
    "upsampling_kernel_size": 5,
    "max_epochs": 300,
    "gpus": [1],
    "log_every_n_steps": 10,
    "check_val_every_n_epoch": 1,
    "sample_images_every_n_epoch": 10,
    "data_dir": "datasets",
    "batch_size": 16,
    "use_perceptual_loss": True,
}


def load_training_parameters(file_path):
    with open(file_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Check and update missing parameters with default values
    for key, value in DEFAULT_PARAMS.items():
        if key not in parameters:
            parameters[key] = value

    # Validate parameter types
    for key, value in parameters.items():
        if key in DEFAULT_PARAMS:
            expected_type = type(DEFAULT_PARAMS[key])
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Invalid type for '{key}' parameter. Expected {expected_type.__name__}, got {type(value)}"
                )

    return parameters
