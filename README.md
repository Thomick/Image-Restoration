 # Image restoration through real noise modeling
 
Model based on the paper :

WAN, Ziyu, ZHANG, Bo, CHEN, Dongdong, et al. Old photo restoration via deep latent space translation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. [Link](https://arxiv.org/abs/2009.07047)

## Structure
- `src/train.py` contains the fonction to set up training of the different modules/models.

- `src/models.py` contains the actual model classes (the 3 modules/models used in the paper are named VAE1, VAE2 and Mapping).

- `src/networks.py` contains the shared network structures

- `src/dataset.py` contains the dataset builders and the dataloaders

- `src/main.py` is used to coordinate the experiments and training sessions

- `src/resize.py` is a small script that downscales all the images in a folder by a certain ratio

- `src/evaluate.py` contains the functions to evaluate the model

Both `train.py` and `evaluate.py` can also be run as scripts to perform the training and evaluation of a model or be imported as modules to be used in other scripts. An example of the usage of the functions can be found at the bottom of the files. Please see the docstring of each function for more information and `README_PRACTICAL.md` for a walkthrough of the training and evaluation process.

## Usage
The code was tested with python 3.8. and torch 1.12.1. The main requirements can be installed with `pip install -r requirements.txt`.
The current version requires to set up the training manually in `src/train.py` and the evaluation in `src/evaluate.py`. The scripts can then be run using the following command:
```python3 src/train.py```
or
```python3 src/evaluate.py```
