# Inference script for the full model.
# Take a folder of images as input and output the denoised images in another folder.
#
# Usage:
#
# python inference.py -i <input_folder> -o <output_folder> -m <fullmodel_path>

from models import Mapping
import torch
from pathlib import Path
import os
from utils import save_image, rescale_colors
from torchvision.transforms import functional as F
from torchvision import io
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = "cpu"


def inference(input_folder, output_folder, fullmodel_path):
    print("Starting denoising...")
    print("Input folder: " + input_folder)
    img_list = os.listdir(input_folder)
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        full_model = Mapping.load_from_checkpoint(
            fullmodel_path,
            inference_mode=True,
            strict=False,
            device=device,
        ).to(device)

        for img_path in img_list:
            img = rescale_colors(
                io.read_image(os.path.join(input_folder, img_path)).float()
            )
            if img.shape[0] == 4:
                img = img[(0, 1, 2), :, :].unsqueeze(0).to(device)
            else:
                img = img.unsqueeze(0).to(device)
            print(img_path + " -> Dimensions :" + str(list(img.shape[-2:])))
            denoised, _, _ = full_model(img)
            save_image(
                denoised.data,
                os.path.join(output_folder, f"{img_path.split('.')[0]}.png"),
            )
            del img
            del denoised
            torch.cuda.empty_cache()
    print("Done!")
    print("Output saved at " + output_folder)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script.")

    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="path to the input folder.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="path to save the output folder.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="path to the model checkpoint.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    inference(args.input_path, args.output_path, args.model_path)
