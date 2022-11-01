# Actual models assembled using the network modules defined in networks.py

import os
import pytorch_lightning as pl
import torchvision.utils as vutils
import torch
from networks import *  # TODO: Import only the required classes
from dataset import random_degradation


# VAE with interwined latent space for real and synthetic images
class VAE1(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAE()
        self.discriminator = Discriminator(4)
        self.discriminator_latent = Discriminator(3, in_channels=64)
        self.params = params
        self.curr_device = None
        self.loss_vgg = VGGLoss_torch()

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device
        opt = self.optimizers()[0]

        # TODO: Use the usual format of batch (data,label)
        input_img, label = batch  # 1=real noise, 0=generated noise
        reconst_img, latent = self(input_img)
        label = label.type_as(input_img).reshape(-1, 1)

        # train VAE1
        if optimizer_idx == 0:
            label_valid = torch.ones(input_img.size(0), 1)
            label_valid = label_valid.type_as(input_img)

            disc_pred = self.discriminator(reconst_img)
            loss_g_gan = F.mse_loss(
                disc_pred, label_valid)
            loss_kl = torch.mean(torch.pow(latent, 2))/2
            loss_reconst = F.l1_loss(reconst_img, input_img)
            loss_latent_gan = F.mse_loss(
                self.discriminator_latent(latent), 1-label)
            vae_loss = loss_kl + \
                self.params["a_reconst"] * loss_reconst + loss_g_gan + \
                loss_latent_gan + \
                self.loss_vgg(reconst_img, input_img)*self.params["lambda_vgg"]

            self.log("vae1_loss", vae_loss, prog_bar=True)
            self.log("loss_g_gan", loss_g_gan, prog_bar=True)
            self.log("loss_latent_gan", loss_latent_gan, prog_bar=True)
            self.log("loss_kl", loss_kl, prog_bar=True)
            self.log("loss_reconst", loss_reconst, prog_bar=True)
            return vae_loss

        # train discriminator
        if optimizer_idx == 1:

            label_valid = torch.ones(input_img.size(0), 1)
            label_valid = label_valid.type_as(input_img)

            real_loss = F.mse_loss(self.discriminator(input_img), label_valid)

            label_fake = torch.zeros(input_img.size(0), 1)
            label_fake = label_fake.type_as(input_img)

            synth_loss = F.mse_loss(
                self.discriminator(reconst_img), label_fake)

            d_loss = (real_loss + synth_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

        # train discriminator for the latent space
        if optimizer_idx == 2:

            d_latent_loss = F.mse_loss(
                self.discriminator_latent(latent), label)

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
        self.curr_device = batch[0].device
        # TODO: Implement the validation step for VAE1
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        #test_input, test_label = batch
        recons = self.vae.decode(self.vae.encode(test_input))
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)
        vutils.save_image(test_input.data,
                          os.path.join(self.logger.log_dir,
                                       "Input",
                                       f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)


class VAE2(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAE()
        self.discriminator = Discriminator()
        self.params = params
        self.curr_device = None

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device

        real_img, _ = batch
        reconst_img, latent = self(real_img)

        # train VAE2
        if optimizer_idx == 0:

            # TODO : Log image in tensorboard
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
            vae_loss = loss_kl + \
                self.params["a_reconst"] * loss_reconst + loss_g_gan
            self.log("vae2_loss", vae_loss, prog_bar=True)
            self.log("loss_g_gan", loss_g_gan, prog_bar=True)
            self.log("loss_kl", loss_kl, prog_bar=True)
            self.log("loss_reconst", loss_reconst, prog_bar=True)
            return vae_loss

        # train discriminator to distinguish real and reconstructed images (1=real, 0=recoonstructed)
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
        self.curr_device = batch[0].device
        # TODO: Implement the validation step for VAE2
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        #test_input, test_label = batch
        recons = self.vae.decode(self.vae.encode(test_input))
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)
        vutils.save_image(test_input.data,
                          os.path.join(self.logger.log_dir,
                                       "Input",
                                       f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)


class Mapping(pl.LightningModule):
    def __init__(self, vae1_encoder, vae2, params, device="cuda"):
        super().__init__()
        self.vae1_encoder = vae1_encoder
        self.vae2 = vae2
        self.mapping = MappingNetwork()
        self.discriminator = Discriminator()
        self.params = params
        self.curr_device = device

    def forward(self, x):
        latent1 = self.vae1_encoder(x)
        latent2 = self.mapping(latent1)
        return self.vae2.decode(latent2), latent1, latent2

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device

        imgs, _ = batch
        clean_img = imgs[:, 0, :, :, :]
        noisy_img = imgs[:, 1, :, :, :]
        if optimizer_idx == 0:
            denoised, _, latent_denoised = self(noisy_img)
            latent_clean = self.vae2.encode(clean_img)
            restored = self.vae2.decode(latent_clean)
            latent_loss = F.l1_loss(latent_denoised, latent_clean)

            label_valid = torch.ones(denoised.size(0), 1).type_as(denoised)
            gan_loss = F.mse_loss(self.discriminator(
                denoised), label_valid)
            mapping_loss = self.params["lambda1_l1"] * latent_loss + gan_loss
            self.log("latent_loss", latent_loss, prog_bar=True)
            self.log("gan_loss", gan_loss, prog_bar=True)
            self.log("mapping_loss", mapping_loss, prog_bar=True)
            return mapping_loss

        # train discriminator to distinguish real and reconstructed images (1=real, 0=recoonstructed)
        if optimizer_idx == 1:

            denoised, _, _ = self(noisy_img)
            restored = self.vae2.decode(self.vae2.encode(clean_img))

            label_valid = torch.ones(denoised.size(0), 1).type_as(denoised)

            real_loss = F.mse_loss(self.discriminator(restored), label_valid)

            label_fake = torch.zeros(denoised.size(0), 1).type_as(denoised)

            fake_loss = F.mse_loss(
                self.discriminator(denoised), label_fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.params['lr']
        b1 = self.params['b1']
        b2 = self.params['b2']

        opt_mapping = torch.optim.Adam(
            self.mapping.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_mapping, opt_d], []

    def validation_step(self, batch, batch_idx):
        self.curr_device = batch[0].device
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        clean_input = test_input[:, 0, :, :, :].to(self.curr_device)
        noisy_input = test_input[:, 1, :, :, :].to(self.curr_device)

        #test_input, test_label = batch
        recons, _, _ = self(noisy_input)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Results",
                                       f"result_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)
        vutils.save_image(clean_input.data,
                          os.path.join(self.logger.log_dir,
                                       "Clean_input",
                                       f"clean_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)
        vutils.save_image(noisy_input.data,
                          os.path.join(self.logger.log_dir,
                                       "Noisy_input",
                                       f"noisy_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          nrow=8)
