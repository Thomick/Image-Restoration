import os
import pytorch_lightning as pl
import torchvision.utils as vutils
import torch.optim as optim
from networks import *


class VAE1(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.generator = VAE()
        self.discriminator = Discriminator()
        self.params = params

        # Activate manual optimization
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        X, _ = batch
        batch_size = X.size(0)
        # TODO: Implement the training step for VAE1
        # 1. Generate fake images
        # 2. Compute the loss for the discriminator
        # 3. Compute the loss for the generator
        # 4. Log the losses
        # 5. Perform the backward pass
        # 6. Perform the optimization step

    def validation_step(self, batch, batch_idx):
        # TODO: Implement the validation step for VAE1
        pass

    def on_validation_end(self):
        self.sample_images()

    def sample_images(self):
        pass

    def configure_optimizers(self):
        # One optimizer for the VAE and one for the feature discriminator
        vae_optim = optim.Adam(self.generator.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.params['LR'],
                                         weight_decay=self.params['weight_decay'])

        return vae_optim, discriminator_optim
