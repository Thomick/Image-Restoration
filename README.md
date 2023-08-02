 # Image restoration through real noise modeling

- [ ] Write a short description of the project before making it public

Architecture base on the paper :

WAN, Ziyu, ZHANG, Bo, CHEN, Dongdong, et al. Old photo restoration via deep latent space translation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. [Link](https://arxiv.org/abs/2009.07047)

## Structure
- `src/train.py` is a script to set up training of the different modules/models. 

- `src/inference.py` is a script that denoises images from a folder using a trained model

- `src/evaluate.py` contains functions to evaluate a trained model (can be modified and used as a script to perform the evaluation of a model)

- `src/models.py` contains the actual model classes (the 3 modules/models used in the paper are named VAE1, VAE2 and Mapping).

- `src/networks.py` contains the shared network structures

- `src/dataset.py` contains the dataset builders and the dataloaders

- `src/utils.py` contains the functions for image processing and evaluation

- `src/config.py` contains the configuration parameters for the training and evaluation


## Usage
The code was tested with python 3.8.0. To install the required packages, run the following command:
```
pip install -r requirements.txt
```
### Training
The model consists in three modules that must be trained separately (VAE1, VAE2 and Mapping). VAE1 and VAE2 can be trained independently, but the mapping requires trained VAE1 and VAE2. The training of the mapping outputs the full model which can then be used for inference. The training of each module is done by running the `train.py` script with the corresponding configuration file (see ```train.py --help```). Here is an example of command for each module:
```
python src/train.py --cfg-path configs/vae.yaml --stage vae1 --output-path checkpoints/vae1.ckpt
python src/train.py --cfg-path configs/vae.yaml --stage vae2 --output-path checkpoints/vae2.ckpt
python src/train.py --cfg-path configs/mapping.yaml --stage mapping --output-path checkpoints/full.ckpt
```
The training can be tracked using tensorboard:
```
tensorboard --logdir lightning_logs
```

### Inference
Given a trained model, the inference can be performed on a folder of images using the `inference.py` script. The script takes as input a folder containing the images to denoise, a folder to save the denoised images and the path to the full model:
```
python src/inference.py -i <input_folder> -o <output_folder> -m <fullmodel_path>
```

Please note that the model is not optimized for handling large images. Running inference on large images may result in out of memory errors.