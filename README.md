 # Image restoration through real noise modeling
 
Model based on the paper :

WAN, Ziyu, ZHANG, Bo, CHEN, Dongdong, et al. Old photo restoration via deep latent space translation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. [Link](https://arxiv.org/abs/2009.07047)

## Structure
- `src/train.py` contains the fonction to set up training of the different modules/models.

- `src/models.py` contains the actual model classes (the 3 modules/models used in the paper are named VAE1, VAE2 and Mapping).

- `src/networks.py` contains the shared network structures

- `src/dataset.py` contains the dataset builders and the dataloaders

- `src/main.py` is used to coordinate the experiments and training sessions