import os
import pytorch_lightning as pl
import torchvision.utils as vutils
import torch.optim as optim
from networks import *

#


class VAE1(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAE()
        self.discriminator = Discriminator()
        self.discriminator_latent = Discriminator()
        self.params = params

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt = self.optimizers()[0]

        # TODO: Use the usual format of batch (data,label)
        real_img, synthetic_img = batch

        # train VAE1
        if optimizer_idx == 0:

            # Loss for real images
            reconst_img, latent = self(real_img)
            label_valid = torch.ones(real_img.size(0), 1)
            label_valid = label_valid.type_as(real_img)

            loss_vae_gan = F.mse_loss(
                self.discriminator(reconst_img), label_valid)
            loss_kl = torch.mean(torch.pow(latent, 2))/2
            loss_reconst = F.l1_loss(reconst_img, real_img)
            vae_loss_real = loss_vae_gan + loss_kl + \
                self.params["a_reconst"] * loss_reconst
            self.log("vae2_loss_real", vae_loss_real, prog_bar=True)

            # Loss for synthetic images
            reconst_img, latent = self(synthetic_img)
            label_valid = torch.ones(real_img.size(0), 1)
            label_valid = label_valid.type_as(real_img)

            loss_vae_gan = F.mse_loss(
                self.discriminator(reconst_img), label_valid)
            loss_kl = torch.mean(torch.pow(latent, 2))/2
            loss_reconst = F.l1_loss(reconst_img, real_img)
            vae_loss_synth = loss_vae_gan + loss_kl + \
                self.params["a_reconst"] * loss_reconst
            self.log("vae2_loss_synth", vae_loss_synth, prog_bar=True)
            return (vae_loss_real + vae_loss_synth)/2

        # train discriminator
        if optimizer_idx == 1:

            label_valid = torch.ones(real_img.size(0), 1)
            label_valid = label_valid.type_as(real_img)

            real_loss = F.mse_loss(self.discriminator(real_img), label_valid)

            label_fake = torch.zeros(real_img.size(0), 1)
            label_fake = label_fake.type_as(real_img)

            synth_loss = F.mse_loss(
                self.discriminator(reconst_img), label_fake)

            d_loss = (real_loss + synth_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

        # train discriminator for the latent space
        if optimizer_idx == 2:
            latent_real = self.vae.encode(real_img)
            latent_synth = self.vae.encode(reconst_img)

            label_valid = torch.ones(latent_real.size(0), 1)
            label_valid = label_valid.type_as(latent_real)

            label_fake = torch.zeros(latent_synth.size(0), 1)
            label_fake = label_fake.type_as(latent_synth)

            real_loss = F.mse_loss(
                self.discriminator_latent(latent_real), label_valid)
            synth_loss = F.mse_loss(
                self.discriminator_latent(latent_synth), label_fake)

            d_latent_loss = (real_loss + synth_loss) / 2
            self.log("d_latent_loss", d_latent_loss, prog_bar=True)
            return d_latent_loss

    def configure_optimizers(self):
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        opt_vae = torch.optim.Adam(
            self.vae.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_d_latent = torch.optim.Adam(
            self.discriminator_latent.parameters(), lr=lr, betas=(b1, b2))
        return [opt_vae, opt_d, opt_d_latent], []

    def validation_step(self, batch, batch_idx):
        # TODO: Implement the validation step for VAE2
        pass


class VAE2(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAE()
        self.discriminator = Discriminator()
        self.params = params

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt = self.optimizers()[0]

        real_img, _ = batch
        reconst_img, latent = self(real_img)

        # train VAE2
        if optimizer_idx == 0:

            # log sampled images
            #sample_imgs = self.generated_imgs[:6]
            #grid = torchvision.utils.make_grid(sample_imgs)
            #self.logger.experiment.add_image("generated_images", grid, 0)

            label_valid = torch.ones(real_img.size(0), 1)
            label_valid = label_valid.type_as(real_img)

            disc_pred = self.discriminator(reconst_img)
            loss_g_gan = F.mse_loss(
                disc_pred, label_valid)
            loss_kl = torch.mean(torch.pow(latent, 2))/2
            loss_reconst = F.l1_loss(reconst_img, real_img)
            vae_loss = loss_g_gan + loss_kl + \
                self.params["a_reconst"] * loss_reconst
            self.log("vae2_loss", vae_loss, prog_bar=True)
            return vae_loss

        # train discriminator
        if optimizer_idx == 1:

            label_valid = torch.ones(real_img.size(0), 1)
            label_valid = label_valid.type_as(real_img)

            real_loss = F.mse_loss(self.discriminator(real_img), label_valid)

            label_fake = torch.zeros(real_img.size(0), 1)
            label_fake = label_fake.type_as(real_img)

            fake_loss = F.mse_loss(
                self.discriminator(reconst_img), label_fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.params['lr']
        b1 = self.params['b1']
        b2 = self.params['b2']

        opt_vae = torch.optim.Adam(
            self.vae.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_vae, opt_d], []

    def validation_step(self, batch, batch_idx):
        # TODO: Implement the validation step for VAE2
        pass
